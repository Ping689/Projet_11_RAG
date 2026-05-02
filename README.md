# Projet 11 - POC RAG OpenAgenda

## Preparation de l'environnement

1. Creer l'environnement virtuel :

```powershell
python -m venv .venv
```

2. Activer l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Installer les dependances :

```powershell
python -m pip install -r requirements.txt
```

4. Configurer les variables d'environnement :

```powershell
Copy-Item .env.example .env
```

5. Verifier les imports :

```powershell
python scripts/check_env.py
```

## Notes

- Pour l'integration Mistral AI, on utilise `mistralai` et `langchain-mistralai`.
- La lecture du fichier `.env` peut se faire avec `python-dotenv`.
- Le package `mistral` mentionne dans certains supports n'est pas le SDK recommande pour l'API Mistral AI.

## Etape 2 - OpenAgenda

1. Renseigner la cle API OpenAgenda dans `.env` :

```env
OPENAGENDA_API_KEY=your_openagenda_api_key_here
OPENAGENDA_AGENDA_UID=your_openagenda_agenda_uid_here
OPENAGENDA_REGION=Ile-de-France
OPENAGENDA_CITY=
OPENAGENDA_LANGUAGE=fr
OPENAGENDA_ALLOWED_CITIES=
```

2. Rechercher un agenda si besoin :

```powershell
python scripts/find_agendas.py --search "Paris culture"
```

3. Recuperer les evenements :

```powershell
python scripts/fetch_openagenda.py
```

4. Pre-traiter les evenements :

```powershell
python scripts/preprocess_events.py
```

5. Lancer les tests de perimetre :

```powershell
python -m pytest tests/test_events_scope.py
```
