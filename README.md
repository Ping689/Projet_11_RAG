# Projet 11 - POC RAG OpenAgenda Puls-Events

**Contexte :** Puls-Events souhaite évaluer la pertinence d'un chatbot capable de recommander des événements culturels à partir de données publiques OpenAgenda. 

**Objectif :** Proposer une interface conversationnelle permettant aux utilisateurs de rechercher des événements en langage naturel, sans utiliser de filtres complexes.

## Étape 1 - Préparation de l'environnement

1. Créer l'environnement virtuel :
```bash
python -m venv .venv
```
2. Activer l'environnement :
```bash
.\.venv\Scripts\Activate.ps1
```
3. Installer les dépendances :
```bash
python -m pip install -r requirements.txt
```
4. Configurer les variables d'environnement :
```bash
Copy-Item .env.example .env
```
5. Vérifier les imports :
```bash
python scripts/check_env.py
```

## Étape 2 - OpenAgenda

1. Renseigner la clé API OpenAgenda dans `.env` :

```env
OPENAGENDA_API_KEY=your_openagenda_api_key_here
OPENAGENDA_AGENDA_UIDS=your_agenda_uid_here
OPENAGENDA_REGION=Ile-de-France
OPENAGENDA_CITY=Paris
OPENAGENDA_SEARCH=culture,concert,exposition,spectacle,musee,theatre,festival
OPENAGENDA_LANGUAGE=fr
```
2. Rechercher un agenda :
```bash
python scripts/find_agendas.py --search "Paris culture"
```
Copier ensuite un ou plusieurs `UID` dans `.env` :

Par exemple :
OPENAGENDA_AGENDA_UIDS=95716291,12345678

3. Récupérer les événements culturels de Paris depuis un ou plusieurs agendas OpenAgenda :
```bash
python scripts/fetch_openagenda.py
```
Le script utilise `GET /v2/agendas/{agendaUID}/events` avec les UID configurés dans `OPENAGENDA_AGENDA_UIDS`.
Il est aussi possible de passer un UID directement :
```bash
python scripts/fetch_openagenda.py --agenda-uid 95716291
```
Pour élargir ou affiner la collecte, modifier `OPENAGENDA_SEARCH` ou passer plusieurs recherches :
```bash
python scripts/fetch_openagenda.py --search "concert" --search "exposition" --search "théâtre"
```
4. Prétraiter les événements :
```bash
python scripts/preprocess_events.py
```
5. Lancer les tests de périmètre :
```bash
python -m pytest tests/test_events_scope.py
```
## Étape 3 - Index FAISS

1. Construire les chunks, calculer les embeddings Mistral et générer l'index FAISS :
```bash
python scripts/build_faiss_index.py
```
2. Les fichiers générés sont écrits dans `data/vector_store/` :

```text
openagenda.faiss
openagenda_metadata.json
```

3. Lancer les tests de l'indexation :
```bash
python -m pytest tests/test_vector_store.py
```
## Étape 4 - Chatbot RAG LangChain

1. Poser une question au chatbot :
```bash
python scripts/chatbot_demo.py --question "Quels événements culturels recommander pour des enfants à Paris ?"
```
2. Lancer le mode interactif :
```bash
python scripts/chatbot_demo.py
```
3. Lancer les tests du pipeline RAG :
```bash
python -m pytest tests/test_rag_chain.py
```
## API de test Q/R

1. Lancer l'API :
```bash
python -m uvicorn app.api:app --reload --port 8001
```
2. Ouvrir la documentation interactive dans le navigateur :
```text
http://127.0.0.1:8001/docs
```
Dans Swagger, ouvrir `POST /ask`, cliquer sur `Try it out`, puis envoyer un corps JSON comme :

```json
{
  "question": "Quels événements culturels recommander pour des enfants à Paris ?",
  "top_k": 3
}
```

Le texte de la question doit rester entre les guillemets sur une seule ligne. Un retour à la ligne non échappé dans `"question"` provoque une erreur `json_invalid`.

3. Tester la même question depuis PowerShell :

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8001/ask `
  -ContentType "application/json" `
  -Body '{"question":"Quels événements culturels recommander pour des enfants à Paris ?","top_k":3}'

4. Vérifier que l'API est démarrée :

```text
http://127.0.0.1:8001/health
```

## Interface Streamlit Q/R

Lancer l'interface web locale :
```bash
python -m streamlit run app/streamlit_app.py
```
Streamlit ouvre ensuite une page locale dans le navigateur :

```text
http://localhost:8501
```
## Note sur le pipeline complet :

```text
fetch_openagenda.py
  ↓
data/raw/openagenda_events_raw.json
  ↓
preprocess_events.py
  ↓
data/processed/openagenda_events_processed.json
  ↓
build_faiss_index.py
  ↓
data/vector_store/openagenda.faiss
data/vector_store/openagenda_metadata.json
  ↓
chatbot_demo.py / API / Streamlit
```

## Phase 3 - Évaluation

Cette phase permet de vérifier la qualité du chatbot RAG à partir d'un jeu de questions/réponses annotées. L'objectif est de contrôler que le système retrouve les bonnes sources dans l'index FAISS et que les réponses générées contiennent les informations attendues.

1. Lancer l'évaluation annotée :
```bash
python scripts/evaluate_rag.py
```
Le script utilise par défaut :

- le jeu d'évaluation : `data/eval/qa_dataset.json`
- l'index vectoriel FAISS : `data/vector_store/`
- le fichier de résultats : `data/eval/rag_eval_results.json`

2. Comprendre le jeu d'évaluation :

Chaque question du fichier `qa_dataset.json` contient :

- `question` : la question posée au chatbot
- `reference_answer` : la réponse attendue
- `expected_event_uids` : les identifiants des événements qui devraient être retrouvés
- `expected_keywords` : les mots-clés qui devraient apparaître dans la réponse

3. Lire les résultats :

Le fichier `rag_eval_results.json` contient les réponses générées et deux indicateurs principaux :

- `source_hit_rate` : proportion de questions pour lesquelles au moins une source attendue a été retrouvée
- `average_keyword_coverage` : proportion moyenne de mots-clés attendus présents dans les réponses

Pour chaque question, on retrouve aussi :

- la question initiale
- la réponse de référence
- la réponse générée par le chatbot
- les sources récupérées depuis FAISS
- le score de similarité de chaque source

4. Interprétation :

Un bon résultat signifie que le RAG retrouve les bons événements et formule une réponse proche de la référence. Si `source_hit_rate` est faible, le problème vient plutôt de la recherche vectorielle ou de l'index FAISS. Si `average_keyword_coverage` est faible, le problème vient plutôt de la génération de réponse ou du prompt.
