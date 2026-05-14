# Rapport technique - POC RAG OpenAgenda Puls-Events

## 1. Resume executif

Puls-Events souhaite evaluer l'interet d'un chatbot capable de recommander des evenements culturels a partir de donnees publiques OpenAgenda. Le besoin metier est de proposer une experience conversationnelle simple : l'utilisateur pose une question en langage naturel, le systeme recherche les evenements pertinents dans une base documentaire, puis genere une reponse synthetique et contextualisee.

Le POC realise couvre la chaine RAG complete : collecte des donnees OpenAgenda, preprocessing, construction d'un texte exploitable pour la recherche semantique, decoupage en chunks, vectorisation avec Mistral, indexation dans FAISS, recherche de similarite, generation de reponse avec Mistral via LangChain, interface CLI, interface API et premiers scripts d'evaluation.

Les resultats montrent que le pipeline technique est fonctionnel de bout en bout. La base vectorielle contient actuellement 1761 chunks issus de 993 evenements pretraites. Le chatbot peut repondre a des questions de recommandation et citer les sources retrouvees dans FAISS. Une evaluation annotee initiale a ete lancee sur 6 questions : le taux de recuperation d'au moins une source attendue est de 0,33 et la couverture moyenne des mots-cles attendus est de 0,61. Ces scores confirment la faisabilite du POC, mais indiquent aussi que le retrieval et le dataset d'evaluation doivent etre ameliores avant une mise en production.

## 2. Contexte et objectifs

Puls-Events gere une plateforme de decouverte d'evenements culturels. Dans ce contexte, un chatbot RAG apporte deux avantages principaux. Le premier est l'acces en langage naturel a des donnees d'evenements, sans imposer a l'utilisateur de comprendre des filtres techniques. Le second est la capacite a produire des recommandations expliquees, c'est-a-dire appuyees sur des sources issues de la base OpenAgenda.

L'objectif du POC n'est pas de livrer un produit final, mais de prouver que l'architecture technique est viable. Le systeme doit pouvoir recuperer des donnees recentes, les structurer, reconstruire la base vectorielle sur demande et generer une reponse augmentee par des resultats de recherche. Le POC doit egalement etre reproductible, documente, testable et presentable en demonstration.

Les livrables associes sont les suivants :

1. Un environnement Python reproductible avec `requirements.txt`.
2. Des scripts de collecte, preprocessing et indexation.
3. Une base FAISS reconstruisible.
4. Un chatbot RAG utilisable en ligne de commande.
5. Une API HTTP pour tester les questions/reponses.
6. Un jeu de donnees annote pour l'evaluation.
7. Un rapport technique et une presentation.

## 3. Perimetre des donnees

La source de donnees utilisee est OpenAgenda. Le POC est desormais configure pour collecter les evenements culturels de Paris depuis la lecture transverse OpenAgenda, sans se limiter a un agenda unique. Lorsque `OPENAGENDA_AGENDA_UID` est vide, `scripts/fetch_openagenda.py` interroge `GET /v2/events` avec les filtres de ville, de region, de langue, de fenetre temporelle et de recherche. Un UID peut encore etre fourni pour revenir a une collecte limitee a un agenda precis.

La configuration cible `OPENAGENDA_REGION=Ile-de-France`, `OPENAGENDA_CITY=Paris` et des recherches culturelles comme `culture`, `concert`, `exposition`, `spectacle`, `musee`, `theatre` et `festival`. La fenetre temporelle par defaut couvre 365 jours en arriere et 365 jours en avant, ce qui permet d'inclure les evenements recents, en cours et a venir, tout en conservant un volume suffisant pour tester la recherche semantique. Les chiffres de volume doivent etre recalcules apres chaque nouvelle collecte.

Les fichiers de donnees sont organises ainsi :

1. `data/raw/openagenda_events_raw.json` : reponse brute OpenAgenda, avec les champs sources.
2. `data/processed/openagenda_events_processed.json` : donnees normalisees et filtrees.
3. `data/vector_store/openagenda.faiss` : index vectoriel FAISS.
4. `data/vector_store/openagenda_metadata.json` : metadonnees des chunks indexes.
5. `data/eval/qa_dataset.json` : questions/reponses annotees pour l'evaluation.
6. `data/eval/rag_eval_results.json` : resultats d'evaluation generes.

## 4. Architecture generale

L'architecture suit un modele RAG classique en deux temps : une phase offline de construction de la base vectorielle, puis une phase online de question/reponse.

La phase offline commence par la collecte OpenAgenda. Le script `scripts/fetch_openagenda.py` interroge soit un agenda precis via `/agendas/{uid}/events`, soit la lecture transverse `/events` lorsque aucun UID n'est configure. Il applique les parametres de region, de ville, de langue, de fenetre temporelle et de recherche, gere la pagination, deduplique les resultats quand plusieurs recherches sont lancees, puis sauvegarde les evenements bruts dans `data/raw/`.

Le preprocessing est ensuite realise par `scripts/preprocess_events.py`. Ce script filtre les timings, applique le perimetre geographique, extrait les champs utiles et construit un champ `text_for_rag`. Ce champ concatene le titre, les descriptions, la ville, la region et les mots-cles. Il sert de representation textuelle principale pour la vectorisation.

La construction de la base vectorielle est assuree par `scripts/build_faiss_index.py`. Ce script charge les evenements pretraites, decoupe `text_for_rag` en chunks, calcule les embeddings avec Mistral, normalise les vecteurs et les indexe dans FAISS. La base peut etre reconstruite a tout moment en relancant cette commande.

La phase online est portee par `app/rag_chain.py`, `scripts/chatbot_demo.py` et `app/api.py`. Pour une question utilisateur, le systeme calcule un embedding de requete, effectue une recherche de similarite dans FAISS, formate les chunks retrouves comme contexte, puis appelle le modele de chat Mistral via une chaine LangChain.

## 5. Description des composants

Le module `app/config.py` centralise la configuration. Il charge les variables d'environnement depuis `.env`, notamment `MISTRAL_API_KEY`, `MISTRAL_EMBEDDING_MODEL`, `MISTRAL_CHAT_MODEL`, `OPENAGENDA_API_KEY` et les parametres OpenAgenda.

Le module `app/openagenda_client.py` encapsule les appels HTTP a OpenAgenda. Il expose la recherche d'agendas, la liste des evenements d'un agenda et la liste transverse des evenements publics indexes. Ce choix isole la logique API et rend les scripts plus lisibles.

Le module `app/vector_store.py` contient la logique de chunking, d'embedding, d'indexation et de recherche. Il definit les structures `EventChunk` et `SearchResult`. La fonction `build_event_chunks` transforme les evenements en chunks, `embed_chunks` calcule les embeddings, `build_faiss_index` cree l'index, `save_vector_store` persiste les fichiers et `similarity_search` interroge FAISS.

Le module `app/rag_chain.py` contient le coeur RAG. Il definit le prompt systeme, formate les sources et orchestre la generation avec LangChain. Le prompt impose au modele de repondre en francais, de rester concis et de s'appuyer uniquement sur le contexte fourni.

Le module `app/api.py` expose une API FastAPI. L'endpoint `GET /health` indique si l'index FAISS est disponible. L'endpoint `POST /ask` accepte une question et un `top_k`, puis retourne la reponse et les sources. Cette API facilite les tests par Swagger, Postman ou une future interface web.

## 6. Choix des modeles

Le modele d'embedding utilise par defaut est `mistral-embed`. Il a ete choisi parce qu'il s'integre directement avec `langchain-mistralai`, qu'il est adapte a la recherche semantique et qu'il evite d'ajouter un second fournisseur de modeles. Les embeddings sont utilises pour representer a la fois les chunks d'evenements et les questions utilisateur dans le meme espace vectoriel.

Le modele de generation utilise par defaut est `mistral-small-latest`. Pour un POC, ce modele presente un bon compromis entre cout, latence et qualite de reponse. La temperature est fixee a `0.2` afin de reduire la variabilite et limiter les hallucinations. Dans le contexte RAG, une temperature faible est preferable, car le modele doit surtout synthese les sources retrouvees plutot qu'inventer des informations.

Le choix de Mistral pour les deux etapes simplifie aussi l'exploitation. Une seule cle API est necessaire, la configuration est claire et la documentation projet reste lisible. Pour une version finale, il serait pertinent de comparer plusieurs modeles d'embedding et de chat sur le dataset d'evaluation afin d'objectiver le compromis qualite/cout.

## 7. Choix FAISS et recherche semantique

FAISS est utilise comme moteur de recherche vectorielle. L'index retenu pour le POC est `IndexFlatIP`. Il s'agit d'un index exact qui calcule le produit scalaire entre la requete et tous les vecteurs indexes. Avant indexation, les vecteurs sont normalises en norme L2. Avec cette normalisation, le produit scalaire devient equivalent a une similarite cosinus.

Ce choix est volontairement simple. Le volume actuel est de 1761 chunks, donc une recherche exhaustive reste rapide et robuste. Un index exact evite les approximations et simplifie l'analyse des resultats pendant la phase POC. Il facilite aussi les tests, car le comportement de retrieval est plus deterministe qu'avec certains index approximatifs.

Pour une mise a l'echelle, `IndexFlatIP` pourrait devenir insuffisant si la base atteint plusieurs centaines de milliers ou millions de chunks. Dans ce cas, FAISS propose des index plus adaptes, comme IVF, HNSW ou PQ. Le choix final dependra du volume, de la latence ciblee, de la memoire disponible et du niveau de rappel attendu.

Le parametre `top_k` controle le nombre de chunks transmis au modele de generation. Par defaut, le POC utilise `top_k=5`. Une valeur trop basse peut manquer des sources utiles. Une valeur trop haute peut ajouter du bruit dans le contexte et degrader la precision de la reponse. Ce parametre devra etre ajuste a partir des resultats d'evaluation.

## 8. Chunking et preparation du texte

Le chunking actuel est base sur le nombre de caracteres, avec une taille par defaut de 900 caracteres et un chevauchement de 120 caracteres. Le chevauchement evite de couper brutalement une information importante a la frontiere entre deux chunks.

Cette approche est suffisante pour un POC, car les evenements OpenAgenda sont generalement courts. Elle permet de limiter la complexite et d'eviter l'ajout d'une dependance supplementaire. Cependant, elle reste moins precise qu'un decoupage par tokens. Pour une version industrialisee, un splitter compatible tokens ou un splitter semantique serait preferable.

Le champ `text_for_rag` est construit a partir des informations les plus utiles a la recommandation : titre, description courte, description longue, ville, region et mots-cles. Cette strategie donne au modele de recherche assez de contexte pour rapprocher une question utilisateur d'un evenement pertinent.

## 9. Integration LangChain

LangChain est utilise pour la partie orchestration de generation. Le projet utilise `ChatPromptTemplate`, `ChatMistralAI` et `StrOutputParser`. La chaine suit le schema suivant : prompt systeme + question utilisateur + contexte FAISS -> modele Mistral -> texte de reponse.

La recherche FAISS est implementee localement plutot que via `langchain-community`. Ce choix reduit le nombre de dependances et garde un controle direct sur le format des metadonnees. Le systeme reste toutefois compatible avec l'esprit LangChain : LangChain orchestre l'appel LLM et le parsing, tandis que FAISS sert de retriever local.

Le prompt demande explicitement au modele de repondre en francais, de ne pas sortir du contexte et de citer les titres, villes et dates quand ces informations existent. Cette contrainte est importante pour limiter les reponses non sourcées.

## 10. Interface de test et demo

Deux interfaces sont disponibles.

La premiere est la demo CLI :

```powershell
python scripts/chatbot_demo.py --question "Quels evenements culturels recommander pour des enfants a Paris ?" --top-k 3
```

Cette commande affiche la reponse ainsi que les sources recuperees depuis FAISS. Elle est pratique pour une demo technique rapide.

La seconde est l'API FastAPI :

```powershell
python -m uvicorn app.api:app --reload
```

L'endpoint `POST /ask` permet de tester une question au format JSON. La documentation interactive est disponible sur `http://127.0.0.1:8000/docs`. Cette interface facilite les tests par une equipe produit ou par une future application front-end.

## 11. Tests et validation

Le projet contient plusieurs tests unitaires.

`tests/test_events_scope.py` verifie que le dataset pretraite existe, qu'il n'est pas vide, que les evenements respectent le perimetre geographique et que les timings sont dans la fenetre attendue.

`tests/test_vector_store.py` valide la creation d'un index FAISS a partir d'embeddings fictifs, puis le rechargement de l'index et des metadonnees.

`tests/test_rag_chain.py` verifie le formatage du contexte et l'orchestration RAG avec un faux modele de chat. Cela permet de tester la logique sans appeler l'API Mistral.

`tests/test_eval_dataset.py` verifie que le dataset d'evaluation contient des questions, reponses attendues, identifiants d'evenements et mots-cles.

`tests/test_api.py` verifie que l'API FastAPI repond correctement sur l'endpoint de sante.

La commande globale est :

```powershell
python -m pytest tests
```

Le dernier lancement a donne 8 tests passes. Les warnings observes concernent le cache `pytest` sous Windows et ne bloquent pas le fonctionnement applicatif.

## 12. Resultats du POC

Le POC a permis de valider les points suivants :

1. Les donnees OpenAgenda peuvent etre collectees et filtrees.
2. Les evenements peuvent etre normalises dans un format exploitable.
3. La base vectorielle FAISS peut etre reconstruite sur demande.
4. Les embeddings Mistral peuvent etre utilises pour indexer les chunks.
5. Une question utilisateur peut etre vectorisee et recherchee dans FAISS.
6. LangChain peut orchestrer la generation d'une reponse augmentee par Mistral.
7. Une interface CLI et une API HTTP permettent de tester le Q/R.
8. Une evaluation annotee initiale peut etre lancee.

La demo reelle suivante a ete executee avec succes :

```powershell
python scripts/chatbot_demo.py --question "Quels evenements culturels recommander pour des enfants a Paris ?" --top-k 3
```

Le systeme doit recommander des evenements jeunesse ou familiaux a Paris et afficher les sources correspondantes avec les titres, villes, dates et scores de similarite. Les exemples exacts dependront de la collecte OpenAgenda disponible au moment de la reconstruction de l'index.

## 13. Evaluation quantitative initiale

Le dataset `data/eval/qa_dataset.json` contient 6 questions annotees. Chaque question comporte une reponse de reference, des identifiants d'evenements attendus et des mots-cles attendus. Le script `scripts/evaluate_rag.py` execute le chatbot sur ce dataset et produit `data/eval/rag_eval_results.json`.

Deux indicateurs sont calcules.

Le premier est `source_hit_rate`. Il mesure la proportion de questions pour lesquelles au moins un evenement attendu est retrouve dans les sources FAISS. Le score actuel est 0,33. Cela signifie que le retriever retrouve une source explicitement attendue pour environ un tiers des questions.

Le second est `average_keyword_coverage`. Il mesure la proportion moyenne de mots-cles attendus presents dans la reponse generee. Le score actuel est 0,61. Ce resultat montre que les reponses contiennent une partie importante du vocabulaire attendu, meme lorsque l'identifiant d'evenement attendu n'est pas toujours retrouve.

Ces resultats doivent etre interpretes avec prudence. Le dataset est encore petit et certaines questions acceptent plusieurs bonnes reponses, alors que les identifiants attendus sont stricts. Par exemple, une question sur le jazz peut retrouver d'autres evenements jazz pertinents que ceux listes initialement. Cela ne signifie pas necessairement que la reponse est mauvaise, mais que l'evaluation doit etre enrichie.

## 14. Limites identifiees

La premiere limite concerne la qualite des donnees sources. Certains champs OpenAgenda sont incomplets, par exemple la region, les coordonnees GPS ou certaines descriptions longues. Cette variabilite peut affecter le preprocessing et la qualite des reponses.

La deuxieme limite concerne le retrieval. Le score `source_hit_rate` de 0,33 montre que la recherche semantique doit etre calibree. Il faudra tester differents `top_k`, enrichir le texte indexe, ajouter des filtres metadonnees et evaluer d'autres strategies de ranking.

La troisieme limite concerne le chunking. Le decoupage par caracteres fonctionne pour le POC, mais il n'est pas optimal pour tous les textes. Un decoupage par tokens ou par sections semantiques pourrait mieux conserver les informations importantes.

La quatrieme limite concerne l'evaluation. Le dataset annote contient seulement 6 questions. Il sert de point de depart, mais il n'est pas suffisant pour conclure sur la qualite globale du chatbot.

Enfin, l'interface API est adaptee au test, mais elle ne comporte pas encore d'authentification, de limitation de debit, de logs metier ou de monitoring des couts API.

## 15. Recommandations pour la suite

La premiere recommandation est d'enrichir le dataset d'evaluation. Il faudrait viser au moins 30 a 50 questions couvrant les cas metier principaux : recommandations par ville, par type d'evenement, par public, par date, par lieu, et questions hors contexte. Chaque question devrait accepter plusieurs sources correctes lorsque le besoin est ouvert.

La deuxieme recommandation est d'ameliorer le retrieval. Plusieurs pistes sont possibles : augmenter `top_k`, enrichir `text_for_rag` avec les dates et lieux de maniere plus normalisee, ajouter un reranking, filtrer par metadonnees lorsque la ville ou la date est explicite, et tester une strategie hybride combinant recherche lexicale et vectorielle.

La troisieme recommandation est d'ajouter une interface web legere pour la demo. L'API existe deja, ce qui facilite la creation d'un front-end simple. Une interface visuelle rendrait la demonstration plus accessible aux equipes produit et marketing.

La quatrieme recommandation est de mettre en place une reconstruction automatisee de l'index. En production, les donnees OpenAgenda evoluent. Il faudrait donc planifier une tache de refresh, stocker les versions d'index et controler la qualite des donnees a chaque reconstruction.

La cinquieme recommandation est d'ajouter du monitoring. Les indicateurs importants sont la latence, le nombre d'appels Mistral, le cout estime, les erreurs API, les questions sans reponse et la satisfaction utilisateur.

## 16. Conclusion

Le POC atteint son objectif principal : demontrer qu'un chatbot RAG base sur OpenAgenda, FAISS, Mistral et LangChain est techniquement realisable. Le pipeline est complet, reproductible et testable. Il dispose maintenant d'une interface CLI, d'une API Q/R, d'un index FAISS, d'un dataset d'evaluation, d'un rapport technique et d'une presentation.

Le niveau de maturite est celui d'un POC fonctionnel. Les fondations techniques sont solides, mais une phase d'amelioration est necessaire avant une version finale. Les priorites sont l'amelioration du retrieval, l'enrichissement de l'evaluation, la creation d'une interface utilisateur et la mise en place d'un suivi operationnel.

En l'etat, le systeme est pret pour une demonstration technique et produit. Il permet de montrer concrètement la valeur du RAG pour la recommandation d'evenements culturels, tout en identifiant clairement les travaux necessaires pour passer a une version industrialisee.
