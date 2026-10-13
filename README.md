# UniAi

**Un modello linguistico costruito da zero.** Niente PyTorch, niente TensorFlow,
niente modelli pre-addestrati: solo Python e NumPy. Ogni pezzo — il calcolo dei
gradienti, la rete neurale, il tokenizer, l'ottimizzatore — è scritto qui, in
circa 800 righe che si possono leggere dall'inizio alla fine.

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

- Backend GPU (CuPy è un sostituto quasi diretto di NumPy)
- KV-cache per una generazione più veloce
- Fine-tuning su coppie domanda/risposta per un vero comportamento da chat
- Mixed precision e gradient accumulation per modelli più grandi
