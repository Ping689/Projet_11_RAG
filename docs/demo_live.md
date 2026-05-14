# Demo live - Chatbot RAG OpenAgenda

## Preparation

1. Activer l'environnement virtuel.
2. Verifier que `.env` contient `MISTRAL_API_KEY`.
3. Reconstruire l'index si besoin :

```powershell
python scripts/build_faiss_index.py
```

## Lancer la demo

Mode question unique :

```powershell
python scripts/chatbot_demo.py --question "Quels evenements culturels recommander pour des enfants a Paris ?" --top-k 3
```

Mode interactif :

```powershell
python scripts/chatbot_demo.py
```

## Questions conseillees

- Quels evenements culturels recommander pour des enfants a Paris ?
- Y a-t-il des activites autour du jazz ?
- Je cherche une exposition a Paris.
- Quels festivals sont proposes a Paris ?

## Points a montrer

La reponse cite des evenements concrets. Les sources affichees sous la reponse montrent les titres, villes, dates et scores de similarite recuperes dans FAISS.
