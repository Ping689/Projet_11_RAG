# Presentation POC RAG Puls-Events

## Slide 1 - Titre

POC Chatbot RAG OpenAgenda pour Puls-Events

## Slide 2 - Contexte

Puls-Events souhaite recommander des evenements culturels recents a partir de donnees OpenAgenda et tester l'interet d'un chatbot intelligent.

## Slide 3 - Objectifs du POC

Valider la collecte OpenAgenda, construire une base vectorielle FAISS, connecter Mistral via LangChain et produire une demo live utilisable.

## Slide 4 - Donnees

Source : OpenAgenda. Perimetre cible : evenements culturels a Paris, collectes via la lecture transverse `/v2/events`. Les volumes sont recalcules apres chaque collecte.

## Slide 5 - Preprocessing

Normalisation des champs utiles, filtrage temporel, filtrage geographique, construction d'un texte RAG par evenement.

## Slide 6 - Vectorisation

Decoupage en chunks, embeddings `mistral-embed`, normalisation L2, indexation dans FAISS.

## Slide 7 - Base FAISS

Index `IndexFlatIP`, similarite cosinus, 1761 chunks indexes, reconstruction possible avec `scripts/build_faiss_index.py`.

## Slide 8 - Architecture RAG

Question utilisateur -> embedding Mistral -> recherche FAISS -> contexte -> prompt LangChain -> ChatMistralAI -> reponse augmentee.

## Slide 9 - Demo live

Commande : `python scripts/chatbot_demo.py`. Exemple : demander des evenements culturels pour enfants a Paris.

## Slide 10 - Evaluation

Dataset annote `data/eval/qa_dataset.json`. Mesures : source hit rate et keyword coverage.

## Slide 11 - Resultats du POC

Pipeline fonctionnel de bout en bout, tests unitaires OK, index FAISS genere, chatbot capable de citer les evenements retrouves.

## Slide 12 - Limites

Qualite variable des donnees OpenAgenda, evaluation encore limitee, pas encore d'interface web, pas de monitoring des couts API.

## Slide 13 - Recommandations

Ajouter une interface demo, etendre le dataset d'evaluation, automatiser la reconstruction, suivre la qualite et les couts.

## Slide 14 - Prochaine etape

Preparer une demo produit plus fluide et tester avec des utilisateurs metier.
