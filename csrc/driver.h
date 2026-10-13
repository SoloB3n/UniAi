/*
 * UniAi — model, training loop, checkpoints and commands.
 *
 * Backend-independent: everything here talks to memory and kernels only
 * through the backend API below, which uniai.c implements for the CPU and
 * uniai_gpu.cu implements for CUDA GPUs.  Pointers named p/g/a/da/d_* live
 * in backend ("device") memory; h_* buffers are on the host.
 *
 * Backend API (defined before this header is included):
 *   void *bk_alloc(size_t bytes)                         zeroed device memory
 *   void  bk_zero(void *p, size_t bytes)
 *   void  bk_upload(void *dst, const void *src, size_t bytes)
 *   void  bk_download(void *dst, const void *src, size_t bytes)
 *   void  bk_describe(char *buf, size_t n)
 *   float bk_grad_norm_clip(float *g, size_t n, float max_norm)
 *   void  bk_adamw(float *p, const float *g, float *m, float *v, size_t n,
 *                  float lr, float wd, int decay, float c1, float c2)
 *   + the layer kernels used in gpt_forward / gpt_backward.
 */

/* ------------------------------------------------------------------ model */

typedef struct { int V, T, L, H, C; } Config;

enum { WTE, WPE, LN1W, LN1B, QKVW, QKVB, APW, APB, LN2W, LN2B,
       FCW, FCB, FPW, FPB, LNFW, LNFB, NPARAMS };

/* activations; each per-layer tensor is stacked over L */
enum { ENC, LN1, LN1M, LN1R, QKV, ATTY, ATT, APROJ, RES2, LN2, LN2M, LN2R,
       FCH, FCHG, FPROJ, RES3, LNF, LNFM, LNFR, PROBS, LOSSES, NACTS };

typedef struct {
    Config cfg;
    size_t psize[NPARAMS], nparams;
    float *params, *grads, *m, *v, *p[NPARAMS], *g[NPARAMS]; /* device */
    float *h_params;                                         /* host mirror */
    int B, T; /* activation capacity */
    size_t asize[NACTS], acts_total;
    float *acts, *dacts, *a[NACTS], *da[NACTS];              /* device */
    int *inputs, *targets;                                   /* host */
    int *d_inputs, *d_targets;                               /* device */
    float *h_row;                                            /* host, V floats */
    int step;
    float val_loss;
} GPT;

static void param_sizes(Config c, size_t *s) {
    size_t C = c.C, L = c.L;
    s[WTE] = (size_t)c.V * C; s[WPE] = (size_t)c.T * C;
    s[LN1W] = s[LN1B] = s[LN2W] = s[LN2B] = L * C;
    s[QKVW] = L * C * 3 * C; s[QKVB] = L * 3 * C;
    s[APW] = L * C * C; s[APB] = L * C;
    s[FCW] = L * C * 4 * C; s[FCB] = L * 4 * C;
    s[FPW] = L * 4 * C * C; s[FPB] = L * C;
    s[LNFW] = s[LNFB] = C;
}

static void gpt_alloc_params(GPT *m, Config c) {
    m->cfg = c;
    param_sizes(c, m->psize);
    m->nparams = 0;
    for (int i = 0; i < NPARAMS; i++) m->nparams += m->psize[i];
    size_t bytes = m->nparams * sizeof(float);
    m->params = (float *)bk_alloc(bytes);
    m->grads = (float *)bk_alloc(bytes);
    m->m = (float *)bk_alloc(bytes);
    m->v = (float *)bk_alloc(bytes);
    m->h_params = (float *)xmalloc(bytes);
    m->h_row = (float *)xmalloc(sizeof(float) * c.V);
    size_t off = 0;
    for (int i = 0; i < NPARAMS; i++) {
        m->p[i] = m->params + off;
        m->g[i] = m->grads + off;
        off += m->psize[i];
    }
}

static void gpt_init(GPT *m) {
    Config c = m->cfg;
    float proj_std = 0.02f / sqrtf(2.0f * c.L);
    float *p = m->h_params;
    for (int i = 0; i < NPARAMS; i++) {
        size_t n = m->psize[i];
        float std = 0.0f, fill = 0.0f;
        switch (i) {
            case WTE: case QKVW: case FCW: std = 0.02f; break;
            case WPE: std = 0.01f; break;
            case APW: case FPW: std = proj_std; break;
            case LN1W: case LN2W: case LNFW: fill = 1.0f; break;
            default: break;
        }
        for (size_t j = 0; j < n; j++) p[j] = std > 0 ? std * rand_normal() : fill;
        p += n;
    }
    bk_upload(m->params, m->h_params, m->nparams * sizeof(float));
}

static void gpt_alloc_acts(GPT *m, int B, int T) {
    Config c = m->cfg;
    size_t L = c.L, N = (size_t)B * T, C = c.C, *s = m->asize;
    s[ENC] = N * C;
    s[LN1] = L * N * C; s[LN1M] = s[LN1R] = L * N;
    s[QKV] = L * N * 3 * C; s[ATTY] = L * N * C;
    s[ATT] = L * B * c.H * (size_t)T * T;
    s[APROJ] = s[RES2] = s[LN2] = L * N * C; s[LN2M] = s[LN2R] = L * N;
    s[FCH] = s[FCHG] = L * N * 4 * C;
    s[FPROJ] = s[RES3] = L * N * C;
    s[LNF] = N * C; s[LNFM] = s[LNFR] = N;
    s[PROBS] = N * c.V; s[LOSSES] = N;
    size_t total = 0;
    for (int i = 0; i < NACTS; i++) total += s[i];
    m->acts_total = total;
    m->acts = (float *)bk_alloc(total * sizeof(float));
    m->dacts = (float *)bk_alloc(total * sizeof(float));
    size_t off = 0;
    for (int i = 0; i < NACTS; i++) {
        m->a[i] = m->acts + off;
        m->da[i] = m->dacts + off;
        off += s[i];
    }
    m->inputs = (int *)xmalloc(N * sizeof(int));
    m->targets = (int *)xmalloc(N * sizeof(int));
    m->d_inputs = (int *)bk_alloc(N * sizeof(int));
    m->d_targets = (int *)bk_alloc(N * sizeof(int));
    m->B = B;
    m->T = T;
}

/* forward over (B,T) tokens in m->inputs; returns the mean loss (if targets).
 * Per-layer slices are offset by the *current* B*T, so smaller batches
 * (e.g. B=1 during sampling) reuse the same buffers. */
static double gpt_forward(GPT *m, int B, int T, int with_targets) {
    Config c = m->cfg;
    int C = c.C, H = c.H;
    size_t N = (size_t)B * T, NC = N * C, att_sz = (size_t)B * H * T * T;
    float **p = m->p, **a = m->a;
    bk_upload(m->d_inputs, m->inputs, N * sizeof(int));
    if (with_targets) bk_upload(m->d_targets, m->targets, N * sizeof(int));
    encoder_forward(a[ENC], m->d_inputs, p[WTE], p[WPE], B, T, C);
    dropout_apply(a[ENC], NC, SITE_EMB);
    for (int l = 0; l < c.L; l++) {
        float *res = l == 0 ? a[ENC] : a[RES3] + (l - 1) * NC;
        float *ln1 = a[LN1] + l * NC, *qkv = a[QKV] + l * 3 * NC, *atty = a[ATTY] + l * NC;
        float *att = a[ATT] + l * att_sz, *aproj = a[APROJ] + l * NC, *res2 = a[RES2] + l * NC;
        float *ln2 = a[LN2] + l * NC, *fch = a[FCH] + l * 4 * NC, *fchg = a[FCHG] + l * 4 * NC;
        float *fproj = a[FPROJ] + l * NC, *res3 = a[RES3] + l * NC;
        layernorm_forward(ln1, a[LN1M] + l * N, a[LN1R] + l * N, res,
                          p[LN1W] + l * C, p[LN1B] + l * C, N, C);
        matmul_forward(qkv, ln1, p[QKVW] + (size_t)l * C * 3 * C, p[QKVB] + l * 3 * C, N, C, 3 * C);
        attention_forward(atty, att, qkv, B, T, C, H, SITE(l, SITE_ATT));
        matmul_forward(aproj, atty, p[APW] + (size_t)l * C * C, p[APB] + l * C, N, C, C);
        dropout_apply(aproj, NC, SITE(l, SITE_APROJ));
        residual_forward(res2, res, aproj, NC);
        layernorm_forward(ln2, a[LN2M] + l * N, a[LN2R] + l * N, res2,
                          p[LN2W] + l * C, p[LN2B] + l * C, N, C);
        matmul_forward(fch, ln2, p[FCW] + (size_t)l * C * 4 * C, p[FCB] + l * 4 * C, N, C, 4 * C);
        gelu_forward(fchg, fch, 4 * NC);
        matmul_forward(fproj, fchg, p[FPW] + (size_t)l * 4 * C * C, p[FPB] + l * C, N, 4 * C, C);
        dropout_apply(fproj, NC, SITE(l, SITE_FPROJ));
        residual_forward(res3, res2, fproj, NC);
    }
    float *last = a[RES3] + (c.L - 1) * NC;
    layernorm_forward(a[LNF], a[LNFM], a[LNFR], last, p[LNFW], p[LNFB], N, C);
    return head_forward(a[PROBS], a[LNF], p[WTE], with_targets ? m->d_targets : NULL,
                        a[LOSSES], N, C, c.V);
}

/* accumulates into m->grads; must follow a forward with targets */
static void gpt_backward(GPT *m, int B, int T) {
    Config c = m->cfg;
    int C = c.C, H = c.H;
    size_t N = (size_t)B * T, NC = N * C, att_sz = (size_t)B * H * T * T;
    float **p = m->p, **g = m->g, **a = m->a, **da = m->da;
    bk_zero(m->dacts, sizeof(float) * m->acts_total);
    float *last = a[RES3] + (c.L - 1) * NC;
    head_backward(da[LNF], g[WTE], a[PROBS], a[LNF], p[WTE], m->d_targets, N, C, c.V);
    layernorm_backward(da[RES3] + (c.L - 1) * NC, g[LNFW], g[LNFB], da[LNF], last,
                       p[LNFW], a[LNFM], a[LNFR], N, C);
    for (int l = c.L - 1; l >= 0; l--) {
        float *res = l == 0 ? a[ENC] : a[RES3] + (l - 1) * NC;
        float *dres = l == 0 ? da[ENC] : da[RES3] + (l - 1) * NC;
        float *ln1 = a[LN1] + l * NC, *qkv = a[QKV] + l * 3 * NC, *atty = a[ATTY] + l * NC;
        float *att = a[ATT] + l * att_sz, *res2 = a[RES2] + l * NC;
        float *ln2 = a[LN2] + l * NC, *fch = a[FCH] + l * 4 * NC, *fchg = a[FCHG] + l * 4 * NC;
        float *dln1 = da[LN1] + l * NC, *dqkv = da[QKV] + l * 3 * NC, *datty = da[ATTY] + l * NC;
        float *daproj = da[APROJ] + l * NC, *dres2 = da[RES2] + l * NC, *dln2 = da[LN2] + l * NC;
        float *dfch = da[FCH] + l * 4 * NC, *dfchg = da[FCHG] + l * 4 * NC;
        float *dfproj = da[FPROJ] + l * NC, *dres3 = da[RES3] + l * NC;
        /* res3 = res2 + fproj */
        accumulate(dres2, dres3, NC);
        accumulate(dfproj, dres3, NC);
        dropout_apply(dfproj, NC, SITE(l, SITE_FPROJ));
        matmul_backward(dfchg, g[FPW] + (size_t)l * 4 * C * C, g[FPB] + l * C, dfproj, fchg,
                        p[FPW] + (size_t)l * 4 * C * C, N, 4 * C, C);
        gelu_backward(dfch, fch, dfchg, 4 * NC);
        matmul_backward(dln2, g[FCW] + (size_t)l * C * 4 * C, g[FCB] + l * 4 * C, dfch, ln2,
                        p[FCW] + (size_t)l * C * 4 * C, N, C, 4 * C);
        layernorm_backward(dres2, g[LN2W] + l * C, g[LN2B] + l * C, dln2, res2,
                           p[LN2W] + l * C, a[LN2M] + l * N, a[LN2R] + l * N, N, C);
        /* res2 = res + aproj */
        accumulate(dres, dres2, NC);
        accumulate(daproj, dres2, NC);
        dropout_apply(daproj, NC, SITE(l, SITE_APROJ));
        matmul_backward(datty, g[APW] + (size_t)l * C * C, g[APB] + l * C, daproj, atty,
                        p[APW] + (size_t)l * C * C, N, C, C);
        attention_backward(dqkv, datty, qkv, att, B, T, C, H, SITE(l, SITE_ATT));
        matmul_backward(dln1, g[QKVW] + (size_t)l * C * 3 * C, g[QKVB] + l * 3 * C, dqkv, ln1,
                        p[QKVW] + (size_t)l * C * 3 * C, N, C, 3 * C);
        layernorm_backward(dres, g[LN1W] + l * C, g[LN1B] + l * C, dln1, res,
                           p[LN1W] + l * C, a[LN1M] + l * N, a[LN1R] + l * N, N, C);
    }
    dropout_apply(da[ENC], NC, SITE_EMB);
    encoder_backward(g[WTE], g[WPE], da[ENC], m->d_inputs, B, T, C);
}

/* identical to uniai/optim.py AdamW: decoupled decay on matrices only */
static void adamw_step(GPT *m, float lr, float wd, int t) {
    float c1 = 1.0f - powf(0.9f, t), c2 = 1.0f - powf(0.95f, t);
    for (int i = 0; i < NPARAMS; i++) {
        int decay = i == WTE || i == WPE || i == QKVW || i == APW || i == FCW || i == FPW;
        size_t off = m->p[i] - m->params;
        bk_adamw(m->p[i], m->g[i], m->m + off, m->v + off, m->psize[i], lr, wd, decay, c1, c2);
    }
}

static float cosine_lr(int step, float max_lr, int warmup, int total) {
    if (step < warmup) return max_lr * (step + 1) / warmup;
    float pr = (float)(step - warmup) / (total - warmup > 1 ? total - warmup : 1);
    if (pr > 1) pr = 1;
    return max_lr * (0.1f + 0.9f * 0.5f * (1.0f + cosf(3.14159265f * pr)));
}

/* --------------------------------------------------------- checkpoint i/o */

/* header: int32[16] = magic, version, V, T, L, H, C, step, has_opt, (float)val_loss */
static void gpt_save(GPT *m, const char *path, int with_opt) {
    int hdr[16] = {MAGIC, VERSION, m->cfg.V, m->cfg.T, m->cfg.L, m->cfg.H, m->cfg.C,
                   m->step, with_opt};
    memcpy(&hdr[9], &m->val_loss, sizeof(float));
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = xfopen(tmp, "wb");
    size_t bytes = m->nparams * sizeof(float);
    fwrite(hdr, sizeof(int), 16, f);
    bk_download(m->h_params, m->params, bytes);
    fwrite(m->h_params, 1, bytes, f);
    if (with_opt) {
        float *buf = (float *)xmalloc(bytes);
        bk_download(buf, m->m, bytes);
        fwrite(buf, 1, bytes, f);
        bk_download(buf, m->v, bytes);
        fwrite(buf, 1, bytes, f);
        free(buf);
    }
    fclose(f);
    rename(tmp, path);
}

static void gpt_load(GPT *m, const char *path) {
    FILE *f = xfopen(path, "rb");
    int hdr[16];
    xfread(hdr, sizeof(int), 16, f);
    if (hdr[0] != MAGIC || hdr[1] != VERSION) { fprintf(stderr, "%s: not a UniAi model\n", path); exit(1); }
    Config c = {hdr[2], hdr[3], hdr[4], hdr[5], hdr[6]};
    gpt_alloc_params(m, c);
    m->step = hdr[7];
    memcpy(&m->val_loss, &hdr[9], sizeof(float));
    size_t bytes = m->nparams * sizeof(float);
    xfread(m->h_params, 1, bytes, f);
    bk_upload(m->params, m->h_params, bytes);
    if (hdr[8]) {
        float *buf = (float *)xmalloc(bytes);
        xfread(buf, 1, bytes, f);
        bk_upload(m->m, buf, bytes);
        xfread(buf, 1, bytes, f);
        bk_upload(m->v, buf, bytes);
        free(buf);
    }
    fclose(f);
}

/* ------------------------------------------------------------------- data */

typedef struct { uint16_t *tok; size_t n; } Tokens;

static Tokens load_tokens(const char *path) {
    FILE *f = xfopen(path, "rb");
    fseek(f, 0, SEEK_END);
    size_t bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    Tokens t = {(uint16_t *)xmalloc(bytes), bytes / 2};
    xfread(t.tok, 2, t.n, f);
    fclose(f);
    return t;
}

static void get_batch(GPT *m, Tokens d, int B, int T) {
    for (int b = 0; b < B; b++) {
        size_t i = ((size_t)rand_u32() << 16 ^ rand_u32()) % (d.n - T - 1);
        for (int t = 0; t < T; t++) {
            m->inputs[b * T + t] = d.tok[i + t];
            m->targets[b * T + t] = d.tok[i + t + 1];
        }
    }
}

/* vocab.bin: int32 V, then per token: uint8 len, bytes */
typedef struct { int V; unsigned char **s; unsigned char *len; } Vocab;

static Vocab load_vocab(const char *path) {
    FILE *f = xfopen(path, "rb");
    Vocab v;
    xfread(&v.V, sizeof(int), 1, f);
    v.s = (unsigned char **)xmalloc(sizeof(char *) * v.V);
    v.len = (unsigned char *)xmalloc(v.V);
    for (int i = 0; i < v.V; i++) {
        xfread(&v.len[i], 1, 1, f);
        v.s[i] = (unsigned char *)xmalloc(v.len[i] + 1);
        xfread(v.s[i], 1, v.len[i], f);
    }
    fclose(f);
    return v;
}

static int sample_probs(const float *p, int V, float temperature) {
    /* probs are softmax(logits); temperature re-weights them as p^(1/temp) */
    double total = 0.0;
    static double *w = NULL;
    static int cap = 0;
    if (cap < V) { free(w); w = (double *)xmalloc(sizeof(double) * V); cap = V; }
    for (int i = 0; i < V; i++) { w[i] = pow(p[i], 1.0 / temperature); total += w[i]; }
    double r = rand_f32() * total, acc = 0.0;
    for (int i = 0; i < V; i++) { acc += w[i]; if (r < acc) return i; }
    return V - 1;
}

static void generate(GPT *m, Vocab *voc, int ntok, float temperature, FILE *out) {
    int T = m->cfg.T, *ctx = (int *)xmalloc(sizeof(int) * (ntok + 1)), n = 1, V = m->cfg.V;
    ctx[0] = '\n';
    for (int k = 0; k < ntok; k++) {
        int len = n < T ? n : T, start = n - len;
        if (len > m->T) { len = m->T; start = n - len; }
        memcpy(m->inputs, ctx + start, sizeof(int) * len);
        gpt_forward(m, 1, len, 0);
        bk_download(m->h_row, m->a[PROBS] + (size_t)(len - 1) * V, sizeof(float) * V);
        int nxt = sample_probs(m->h_row, V, temperature);
        ctx[n++] = nxt;
        if (voc) fwrite(voc->s[nxt], 1, voc->len[nxt], out);
    }
    fflush(out);
    free(ctx);
}

/* ------------------------------------------------------------------- main */

static void usage(void) {
    fprintf(stderr,
        "UniAi engine (%s backend)\n"
        "  train  -i PREFIX -o OUT.bin [-r RESUME.bin] [-L layers] [-H heads]\n"
        "         [-C embd] [-T block] [-B batch] [-n steps] [-l lr] [-w warmup]\n"
        "         [-d weight_decay] [-p dropout] [-e eval_every] [-s seed]\n"
        "         reads PREFIX_train.bin, PREFIX_val.bin, PREFIX_vocab.bin\n"
        "  sample -m MODEL.bin -v VOCAB.bin [-n tokens] [-t temperature] [-s seed]\n"
        "  check  MODEL.bin BATCH.bin OUT.bin\n"
        "  gradcheck MODEL.bin BATCH.bin DROPOUT\n", BACKEND_NAME);
    exit(1);
}

static float eval_loss(GPT *m, Tokens d, int B, int T, int iters) {
    double s = 0.0;
    for (int i = 0; i < iters; i++) {
        get_batch(m, d, B, T);
        s += gpt_forward(m, B, T, 1);
    }
    return (float)(s / iters);
}

static int cmd_train(int argc, char **argv) {
    const char *prefix = NULL, *out = "model.bin", *resume = NULL;
    Config c = {0, 128, 4, 4, 128};
    int B = 16, steps = 3000, warmup = 100, eval_every = 500, eval_iters = 20;
    float lr = 1e-3f, wd = 0.1f, dropout = 0.0f;
    for (int i = 2; i + 1 < argc; i += 2) {
        char f = argv[i][1];
        const char *v = argv[i + 1];
        if (f == 'i') prefix = v; else if (f == 'o') out = v; else if (f == 'r') resume = v;
        else if (f == 'L') c.L = atoi(v); else if (f == 'H') c.H = atoi(v);
        else if (f == 'C') c.C = atoi(v); else if (f == 'T') c.T = atoi(v);
        else if (f == 'B') B = atoi(v); else if (f == 'n') steps = atoi(v);
        else if (f == 'l') lr = atof(v); else if (f == 'w') warmup = atoi(v);
        else if (f == 'd') wd = atof(v); else if (f == 'e') eval_every = atoi(v);
        else if (f == 'p') dropout = atof(v);
        else if (f == 's') rng_state = strtoull(v, NULL, 10) * 2654435761ULL + 1;
        else usage();
    }
    if (!prefix) usage();
    char path[1024];
    snprintf(path, sizeof path, "%s_train.bin", prefix);
    Tokens train = load_tokens(path);
    snprintf(path, sizeof path, "%s_val.bin", prefix);
    Tokens val = load_tokens(path);
    snprintf(path, sizeof path, "%s_vocab.bin", prefix);
    Vocab voc = load_vocab(path);

    GPT m;
    memset(&m, 0, sizeof m);
    if (resume) {
        gpt_load(&m, resume);
        printf("resuming %s at step %d\n", resume, m.step);
    } else {
        c.V = voc.V;
        if (c.C % c.H) { fprintf(stderr, "embd must be divisible by heads\n"); return 1; }
        gpt_alloc_params(&m, c);
        gpt_init(&m);
        m.val_loss = 1e30f;
    }
    c = m.cfg;
    int T = c.T;
    gpt_alloc_acts(&m, B, T);
    char desc[1024];
    bk_describe(desc, sizeof desc);
    printf("UniAi %s engine | %zu parameters | V=%d T=%d L=%d H=%d C=%d | batch %d | %s\n",
           BACKEND_NAME, m.nparams, c.V, c.T, c.L, c.H, c.C, B, desc);
    printf("train %zu tokens, val %zu tokens\n", train.n, val.n);

    double t0 = now(), tlast = t0;
    for (int step = m.step; step < steps; step++) {
        float cur_lr = cosine_lr(step, lr, warmup, steps);
        get_batch(&m, train, B, T);
        drop_p = dropout;
        drop_seed = ((uint64_t)rand_u32() << 32) | rand_u32();
        float loss = (float)gpt_forward(&m, B, T, 1);
        bk_zero(m.grads, m.nparams * sizeof(float));
        gpt_backward(&m, B, T);
        drop_p = 0.0f; /* evaluation and sampling run without dropout */
        float norm = bk_grad_norm_clip(m.grads, m.nparams, 1.0f);
        adamw_step(&m, cur_lr, wd, step + 1);
        m.step = step + 1;
        if (step % 50 == 0) {
            double t = now();
            printf("step %5d | loss %.4f | lr %.2e | grad %.2f | %.1f ms/step\n", step, loss,
                   cur_lr, norm, (t - tlast) * 1000.0 / (step > 0 ? 50 : 1));
            tlast = t;
            fflush(stdout);
        }
        if ((step + 1) % eval_every == 0 || step + 1 == steps) {
            float tl = eval_loss(&m, train, B, T, eval_iters);
            float vl = eval_loss(&m, val, B, T, eval_iters);
            printf("---- eval @ %d: train %.4f | val %.4f | val ppl %.1f | %.0fs elapsed\n",
                   step + 1, tl, vl, expf(vl), now() - t0);
            if (vl < m.val_loss) {
                m.val_loss = vl;
                gpt_save(&m, out, 1);
                printf("     saved %s\n", out);
            }
            printf("     sample: ");
            generate(&m, &voc, 80, 0.8f, stdout);
            printf("\n");
            fflush(stdout);
            tlast = now(); /* keep eval time out of the ms/step figure */
        }
    }
    printf("done in %.1fs | best val loss %.4f\n", now() - t0, m.val_loss);
    return 0;
}

static int cmd_sample(int argc, char **argv) {
    const char *model = NULL, *vocab = NULL;
    int n = 300;
    float temp = 0.8f;
    rng_state = (uint64_t)time(NULL);
    for (int i = 2; i + 1 < argc; i += 2) {
        char f = argv[i][1];
        const char *v = argv[i + 1];
        if (f == 'm') model = v; else if (f == 'v') vocab = v;
        else if (f == 'n') n = atoi(v); else if (f == 't') temp = atof(v);
        else if (f == 's') rng_state = strtoull(v, NULL, 10) * 2654435761ULL + 1;
        else usage();
    }
    if (!model || !vocab) usage();
    GPT m;
    memset(&m, 0, sizeof m);
    gpt_load(&m, model);
    Vocab voc = load_vocab(vocab);
    gpt_alloc_acts(&m, 1, m.cfg.T);
    generate(&m, &voc, n, temp, stdout);
    printf("\n");
    return 0;
}

static void load_batch(GPT *m, const char *path, int *B, int *T) {
    FILE *f = xfopen(path, "rb");
    int hdr[2];
    xfread(hdr, sizeof(int), 2, f);
    *B = hdr[0];
    *T = hdr[1];
    gpt_alloc_acts(m, *B, *T);
    xfread(m->inputs, sizeof(int), (size_t)*B * *T, f);
    xfread(m->targets, sizeof(int), (size_t)*B * *T, f);
    fclose(f);
}

/* forward + backward + one AdamW step on a fixed batch; dumps everything */
static int cmd_check(int argc, char **argv) {
    if (argc != 5) usage();
    GPT m;
    memset(&m, 0, sizeof m);
    gpt_load(&m, argv[2]);
    int B, T;
    load_batch(&m, argv[3], &B, &T);
    float loss = (float)gpt_forward(&m, B, T, 1);
    gpt_backward(&m, B, T);
    size_t bytes = m.nparams * sizeof(float);
    float *buf = (float *)xmalloc(bytes);
    FILE *f = xfopen(argv[4], "wb");
    fwrite(&loss, sizeof(float), 1, f);
    bk_download(buf, m.grads, bytes);
    fwrite(buf, 1, bytes, f);
    float norm = bk_grad_norm_clip(m.grads, m.nparams, 1.0f);
    adamw_step(&m, 1e-3f, 0.1f, 1);
    fwrite(&norm, sizeof(float), 1, f);
    bk_download(buf, m.params, bytes);
    fwrite(buf, 1, bytes, f);
    fclose(f);
    free(buf);
    return 0;
}

/* finite-difference check of the analytic gradients with dropout active:
 * same seed => same masks, so the loss is a deterministic function */
static int cmd_gradcheck(int argc, char **argv) {
    if (argc != 5) usage();
    GPT m;
    memset(&m, 0, sizeof m);
    gpt_load(&m, argv[2]);
    int B, T;
    load_batch(&m, argv[3], &B, &T);
    drop_p = atof(argv[4]);
    drop_seed = 12345;
    gpt_forward(&m, B, T, 1);
    gpt_backward(&m, B, T);
    float *grads = (float *)xmalloc(m.nparams * sizeof(float));
    bk_download(grads, m.grads, m.nparams * sizeof(float));
    double worst = 0.0;
    size_t off = 0;
    for (int i = 0; i < NPARAMS; i++) {
        size_t best = 0;  /* probe the entry with the largest gradient */
        for (size_t j = 0; j < m.psize[i]; j++)
            if (fabsf(grads[off + j]) > fabsf(grads[off + best])) best = j;
        float old = m.h_params[off + best], eps = 1e-2f, val;
        val = old + eps;
        bk_upload(m.p[i] + best, &val, sizeof(float));
        double lp = gpt_forward(&m, B, T, 1);
        val = old - eps;
        bk_upload(m.p[i] + best, &val, sizeof(float));
        double lm = gpt_forward(&m, B, T, 1);
        bk_upload(m.p[i] + best, &old, sizeof(float));
        double num = (lp - lm) / (2 * eps), ana = grads[off + best];
        double err = fabs(num - ana) / (fabs(num) + fabs(ana) + 1e-8);
        if (err > worst) worst = err;
        printf("param %2d  analytic % .6e  numeric % .6e  rel err %.1e\n", i, ana, num, err);
        off += m.psize[i];
    }
    printf("worst rel err %.1e\n", worst);
    free(grads);
    return worst < 2e-2 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    if (!strcmp(argv[1], "train")) return cmd_train(argc, argv);
    if (!strcmp(argv[1], "sample")) return cmd_sample(argc, argv);
    if (!strcmp(argv[1], "check")) return cmd_check(argc, argv);
    if (!strcmp(argv[1], "gradcheck")) return cmd_gradcheck(argc, argv);
    usage();
    return 1;
}
