# UniAi - AGI Destination

**Un modello linguistico costruito da zero.** Niente PyTorch, niente TensorFlow,
niente BLAS, niente modelli pre-addestrati. Ogni pezzo — il calcolo dei
gradienti, la rete neurale, il tokenizer, l'ottimizzatore, perfino la
moltiplicazione di matrici — è scritto qui.

Ci sono **due motori** che implementano lo stesso identico modello:

- **Python + NumPy** (`uniai/`): il laboratorio. Autograd generico, leggibile,
  ogni gradiente verificato numericamente.
- **C puro** (`csrc/uniai.c`): il motore da corsa. Forward e backward scritti
  a mano kernel per kernel, un GEMM proprio con tiling a registri, OpenMP su
  tutti i core, zero dipendenze oltre a `libm`.

I due motori sono verificati l'uno contro l'altro: stessi pesi + stesso batch
⇒ stessa loss, stesso gradiente per ogni parametro e stesso passo di AdamW
(errore relativo ~10⁻⁷).

È la stessa architettura dei GPT (un *transformer decoder*), in miniatura.
Gli dai un testo, lui impara a scrivere nello stesso stile.

```
testo ──► tokenizer BPE ──► GPT (attention + MLP) ──► probabilità del token successivo
                                    ▲                            │
                                    └──── autograd + AdamW ◄─────┘ (errore)
```

## Cosa c'è dentro

| File | Cosa fa |
|---|---|
| `uniai/tensor.py` | **Motore di autodifferenziazione.** Un `Tensor` ricorda come è stato calcolato; `.backward()` percorre il grafo all'indietro e calcola i gradienti (matmul batch, softmax, GELU, LayerNorm e cross-entropy fusi, broadcasting). |
| `uniai/nn.py` | **La rete.** `Linear`, `Embedding`, `LayerNorm`, self-attention causale multi-testa, blocchi transformer pre-norm, il modello `GPT` con weight tying, e il campionamento (temperatura, top-k, top-p). |
| `uniai/tokenizer.py` | **Tokenizer BPE a livello di byte**, addestrato sul tuo testo. Qualsiasi stringa (accenti, emoji) va e torna senza perdite. |
| `uniai/optim.py` | AdamW, clipping del gradiente, learning rate con warmup + coseno. |
| `uniai/checkpoint.py` | Modello + tokenizer in un unico file `.npz`. |
| `uniai/__main__.py` | La riga di comando: `train`, `sample`, `chat`. |
| `tests/test_autograd.py` | Ogni gradiente è verificato contro le differenze finite — anche quello dell'intero transformer. |
| `csrc/uniai.c` | **Il motore in C.** Encoder, LayerNorm, attention causale, GELU, testa con pesi condivisi, AdamW: forward e backward scritti a mano. Memoria allocata una sola volta. |
| `uniai/cformat.py` | Formati binari condivisi tra Python e C (modello, token, vocabolario). |
| `tests/test_c_engine.py` | Test di parità C ↔ Python: loss, gradienti e passo dell'ottimizzatore. |

## Provalo

```sh
pip install numpy
./data/download.sh                      # oppure usa un tuo .txt qualsiasi
python -m uniai train --data data/shakespeare.txt --out uniai.npz
python -m uniai chat  --ckpt uniai.npz
```

`chat` è un completamento interattivo: scrivi l'inizio di un testo e UniAi lo
continua. Altri comandi:

```sh
python -m uniai sample --ckpt uniai.npz --prompt "ROMEO:" --tokens 400
python -m uniai train  --resume uniai.npz --data data/shakespeare.txt --steps 6000
python tests/test_autograd.py
```

### Il motore in C

```sh
make                                                    # compila ./uniai_c
python -m uniai prepare --data data/shakespeare.txt --out data/shakespeare
./uniai_c train -i data/shakespeare -o model.bin        # addestra in C
python -m uniai chat --ckpt model.bin --tokenizer data/shakespeare_tokenizer.json
make test                                               # gradcheck + parità C/Python
```

Python prepara i dati (tokenizer BPE) e offre la chat; il C fa il lavoro
pesante. Opzioni di `uniai_c train`: `-L` strati, `-H` teste, `-C` dimensione,
`-T` contesto, `-B` batch, `-n` passi, `-l` learning rate, `-p` dropout,
`-r` per riprendere.

#### Risultati misurati

Stessa macchina (4 core Xeon, AVX-512), stesso modello (0,94M parametri, 4
strati, contesto 128, vocabolario BPE 1024), batch 16, corpus Tiny
Shakespeare (1,1 MB):

| | Python + NumPy | C |
|---|---|---|
| Tempo per passo | ~900 ms | **~160 ms (≈5,6× più veloce)** |
| 3000 passi | ~45 min | **8 min 44 s** |
| GEMM (kernel singolo) | — | 220–240 GFLOP/s |

| Addestramento C, 3000 passi | val loss | perplessità |
|---|---|---|
| senza dropout | **3,50** | 33 |
| dropout 0,1 (`-p 0.1`) | 3,62 | 37 |

Su un addestramento così breve il dropout rallenta l'apprendimento più di
quanto riduca l'overfitting, quindi in C il default è 0; diventa utile con
più passi o modelli più grandi rispetto ai dati.

Esempio generato dal modello C (prompt `ROMEO:`):

```
ROMEO:
Why, my lord?

ROMEO:
I do not fight against his daughter,
...
HENRY BOLINGBROKE:
I will, madam: I hope, if thou art crember,
```

Ha imparato la forma di un'opera teatrale (personaggi, versi, dialoghi) e
molto vocabolario; il senso delle frasi richiede modelli e dati più grandi.

Il dropout nel motore C non salva maschere: ogni decisione tieni/scarta è un
hash di (seme del passo, punto della rete, indice), e il backward la rigenera
identica. `uniai_c gradcheck` lo verifica con le differenze finite.

Come è fatto il GEMM: un blocco di 6 righe × 32 colonne (16 senza AVX-512) del
risultato resta nei registri per tutto il ciclo su K, così ogni dato caricato
viene riusato 6 volte; con `-march=native` il compilatore lo traduce in
istruzioni FMA vettoriali. Una variante legge A trasposta direttamente, così
il gradiente dei pesi `dW = Xᵀ·dY` non richiede trasposizioni esplicite.

### Addestralo su ciò che vuoi

Funziona con **qualunque testo UTF-8**: la Divina Commedia, i tuoi appunti
dell'università, le lettere di famiglia, codice Python. Più testo = risultati
migliori. Per un modello più grande:

```sh
python -m uniai train --data mio_testo.txt --layers 6 --heads 6 --embd 192 \
                      --block 256 --vocab 2048 --steps 10000
```

Su CPU, il modello predefinito (~0,9M parametri) impara la struttura di
Shakespeare (nomi dei personaggi, versi, punteggiatura, vocabolario) in pochi
minuti di addestramento.

## Onestà sui limiti

UniAi è vero deep learning, ma su scala da laptop: qualche milione di parametri
contro le centinaia di miliardi dei grandi modelli, che vengono addestrati su
migliaia di GPU con trilioni di parole. Non "capisce" e non ragiona: impara le
regolarità statistiche del testo che gli dai. Proprio per questo è prezioso —
è abbastanza piccolo da essere capito per intero, riga per riga.

## Prossimi passi possibili

- Kernel CUDA per il motore in C (la struttura kernel-per-kernel è già quella giusta)
- Flash attention (attention a blocchi senza materializzare la matrice T×T)
- KV-cache per una generazione più veloce
- Fine-tuning su coppie domanda/risposta per un vero comportamento da chat
- Mixed precision e gradient accumulation per modelli più grandi
