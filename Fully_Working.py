import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
from collections import deque, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
import os
import requests
import re
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize # Import word_tokenize
import nltk
import matplotlib.pyplot as plt
import torch.optim as optim
from random import uniform


warnings.filterwarnings('ignore')
def temperature_scale(logits, temperature):
    return logits / temperature

import numpy as np

def expected_calibration_error(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == (probs[in_bin] >= 0.5))
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

# --- Step 1: Data Cleaning and Preprocessing ---
import os
import re
import pickle
import requests

def download_war_and_peace():
    """Download War and Peace from Project Gutenberg"""
    url = "https://www.gutenberg.org/files/2600/2600-0.txt"
    filename = "war_and_peace.txt"

    if os.path.exists(filename):
        print(f"📄 Using cached copy: {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    print(f"🌐 Downloading from {url}")
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("✅ Downloaded and saved to war_and_peace.txt")
        return response.text
    else:
        raise Exception("❌ Failed to download War and Peace.")

def clean_text(text):
    """Clean the downloaded text from Gutenberg headers/footers and formatting."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
        text = text[text.find('\n') + 1:]

    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'CHAPTER [IVXLC]+\.?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'BOOK [IVXLC]+\.?\n', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_paragraphs(text, min_length=100, max_paragraphs=1000):
    """Extract meaningful paragraphs from the text"""
    paragraphs = text.split('\n\n')
    filtered_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if (len(para) >= min_length and
            not para.isupper() and
            len(para.split()) >= 10):
            filtered_paragraphs.append(para)
    return filtered_paragraphs[:max_paragraphs]

def preprocess_data_if_needed():
    """Main preprocessing function to run if data is not found"""
    if not os.path.exists('war_and_peace_paragraphs.pkl'):
        print("⚙️ Preprocessing War and Peace...")
        raw_text = download_war_and_peace()
        cleaned_text = clean_text(raw_text)
        paragraphs = extract_paragraphs(cleaned_text)
        with open('war_and_peace_paragraphs.pkl', 'wb') as f:
            pickle.dump(paragraphs, f)
        print("✅ Preprocessing complete! Saved to 'war_and_peace_paragraphs.pkl'")
    else:
        print("✅ Preprocessed data found. Skipping preprocessing.")
# --- Step 2: Train-Test Split and Sentence Tokenization ---
def tokenize_sentences(paragraphs):
    """Tokenize paragraphs into sentences and create dataset"""
    dataset = []
    for para_idx, paragraph in enumerate(paragraphs):
        sentences = sent_tokenize(paragraph)
        valid_sentences = []
        for sent in sentences:
            if len(sent.strip()) > 20 and len(sent.split()) >= 3:
                valid_sentences.append(sent.strip())
        if len(valid_sentences) >= 2:
            dataset.append({
                'paragraph_id': para_idx,
                'paragraph': paragraph,
                'sentences': valid_sentences,
                'num_sentences': len(valid_sentences)
            })
    return dataset

def create_sentence_pairs(dataset):
    """Create sentence pairs for boundary detection training"""
    sentence_data = []
    for item in dataset:
        para_id = item['paragraph_id']
        sentences = item['sentences']
        for sent_idx, sentence in enumerate(sentences):
            sentence_data.append({
                'paragraph_id': para_id,
                'sentence_id': sent_idx,
                'sentence': sentence,
                'is_first_sentence': sent_idx == 0,
                'is_last_sentence': sent_idx == len(sentences) - 1,
                'sentence_position': sent_idx,
                'total_sentences': len(sentences)
            })
    return sentence_data

def split_data_if_needed():
    """Split data into train/test if not already done"""
    if not os.path.exists('train_data.pkl') or not os.path.exists('test_data.pkl'):
        print("Splitting data into train/test sets...")
        with open('war_and_peace_paragraphs.pkl', 'rb') as f:
            paragraphs = pickle.load(f)

        # ✅ Reduce dataset size
        paragraphs = paragraphs[:12000]
        print(f"🔎 Paragraphs loaded: {len(paragraphs)}")  # ✅ Debug print

        # ✅ Tokenize and check output
        dataset = tokenize_sentences(paragraphs)
        print(f"🔎 Sentence tokens generated: {len(dataset)}")  # ✅ Debug print

        # ✅ Fail-safe
        if len(dataset) == 0:
            raise ValueError("❌ No tokenized sentences found! Check your preprocessing or paragraph filtering.")

        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            dataset, test_size=0.2, random_state=42
        )

        train_sentences = create_sentence_pairs(train_data)
        test_sentences = create_sentence_pairs(test_data)

        # ✅ Limit sentence pair count
        train_sentences = train_sentences[:100000]
        test_sentences = test_sentences[:10000]

        with open('train_data.pkl', 'wb') as f:
            pickle.dump({'paragraphs': train_data, 'sentences': train_sentences}, f)
        with open('test_data.pkl', 'wb') as f:
            pickle.dump({'paragraphs': test_data, 'sentences': test_sentences}, f)

        print("✅ Data split complete! Files saved: train_data.pkl, test_data.pkl")
    else:
        print("✅ Train/test data found. Skipping splitting.")


# --- Step 3: SVO Triplet Extraction using spaCy ---
def extract_svo_triplets(sentence, nlp_model):
    """Extract Subject-Verb-Object triplets from a sentence"""
    doc = nlp_model(sentence)
    triplets = []
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"]:
            verb = token.lemma_.lower()
            subject = None
            objects = []

            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = extract_noun_phrase(child)
                    break

            for child in token.children:
                if child.dep_ in ["dobj", "iobj", "pobj"]:
                    objects.append(extract_noun_phrase(child))
                elif child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            objects.append(extract_noun_phrase(grandchild))

            if subject and objects:
                for obj in objects:
                    if subject != obj:
                        triplets.append({
                            'subject': subject, 'verb': verb, 'object': obj, 'sentence': sentence
                        })
            elif subject and not objects:
                triplets.append({
                    'subject': subject, 'verb': verb, 'object': f"ACTION_{verb}", 'sentence': sentence
                })
    return triplets

def extract_noun_phrase(token):
    """Extract noun phrase including modifiers"""
    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
        phrase_parts = []
        for child in token.lefts:
            if child.pos_ in ["ADJ", "DET", "NUM"]:
                phrase_parts.append(child.text.lower())
        phrase_parts.append(token.text.lower())
        for child in token.rights:
            if child.pos_ in ["ADJ", "NOUN"]:
                phrase_parts.append(child.text.lower())
        return "_".join(phrase_parts)
    return token.text.lower()

def process_sentences_to_triplets(sentences_data, nlp_model):
    """Process all sentences to extract SVO triplets"""
    all_triplets = []
    sentence_triplet_map = {}
    for sent_data in sentences_data:
        para_id = sent_data['paragraph_id']
        sent_id = sent_data['sentence_id']
        sentence = sent_data['sentence']
        triplets = extract_svo_triplets(sentence, nlp_model)
        for triplet in triplets:
            triplet.update({
                'paragraph_id': para_id,
                'sentence_id': sent_id,
                'unique_sentence_id': f"{para_id}_{sent_id}"
            })
        all_triplets.extend(triplets)
        sentence_triplet_map[f"{para_id}_{sent_id}"] = triplets
    return all_triplets, sentence_triplet_map

def create_vocabulary(triplets):
    """Create vocabulary of all entities (subjects and objects)"""
    entities = set()
    verbs = set()
    for triplet in triplets:
        entities.add(triplet['subject'])
        entities.add(triplet['object'])
        verbs.add(triplet['verb'])
    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}
    verb_to_id = {verb: idx for idx, verb in enumerate(sorted(verbs))}
    return entity_to_id, verb_to_id

def extract_svo_if_needed():
    """Extract SVO triplets if not already done"""
    if not os.path.exists('svo_triplets.pkl'):
        print("Extracting SVO triplets...")
        nlp = spacy.load("en_core_web_sm")
        with open('train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)

        train_triplets, train_sentence_map = process_sentences_to_triplets(train_data['sentences'], nlp)
        test_triplets, test_sentence_map = process_sentences_to_triplets(test_data['sentences'], nlp)

        all_triplets = train_triplets + test_triplets
        entity_to_id, verb_to_id = create_vocabulary(all_triplets)

        processed_data = {
            'train_triplets': train_triplets,
            'test_triplets': test_triplets,
            'train_sentence_map': train_sentence_map,
            'test_sentence_map': test_sentence_map,
            'entity_to_id': entity_to_id,
            'verb_to_id': verb_to_id
        }
        with open('svo_triplets.pkl', 'wb') as f:
            pickle.dump(processed_data, f)
        print("SVO extraction complete! Data saved to 'svo_triplets.pkl'")
    else:
        print("SVO triplets found. Skipping extraction.")

# --- Step 4: Knowledge Graph Creation ---
class NetworkXGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.sentence_entities = {}

    def create_graph_from_triplets(self, triplets):
        for triplet in triplets:
            subject = triplet['subject']
            obj = triplet['object']
            verb = triplet['verb']
            sent_id = triplet['unique_sentence_id']
            para_id = triplet['paragraph_id']

            self.graph.add_node(subject, sentence_id=sent_id, paragraph_id=para_id, entity_type='entity')
            self.graph.add_node(obj, sentence_id=sent_id, paragraph_id=para_id, entity_type='entity')
            self.graph.add_edge(subject, obj, verb=verb, sentence_id=sent_id, paragraph_id=para_id, weight=1.0)

            if sent_id not in self.sentence_entities:
                self.sentence_entities[sent_id] = set()
            self.sentence_entities[sent_id].add(subject)
            self.sentence_entities[sent_id].add(obj)

    def get_neighbors(self, node, max_depth=2):
        if node not in self.graph:
            return set()
        neighbors = set([node])
        current_level = set([node])
        for _ in range(max_depth):
            next_level = set()
            for n in current_level:
                next_level.update(self.graph.predecessors(n))
                next_level.update(self.graph.successors(n))
            current_level = next_level - neighbors
            neighbors.update(current_level)
            if not current_level:
                break
        return neighbors

    def save_graph(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.graph, f)

    def load_graph(self, filename):
        with open(filename, 'rb') as f:
            self.graph = pickle.load(f)

def create_boundary_training_data(triplets, sentence_map, nx_manager):
    boundary_data = []
    COMMON_WORDS = {"the", "a", "an", "he", "she", "it", "is", "this", "that"}

    for sent_id, sent_triplets in sentence_map.items():
        if not sent_triplets:
            continue

        entities = set()
        for triplet in sent_triplets:
            entities.add(triplet['subject'])
            entities.add(triplet['object'])

        for entity in entities:
            # ✅ Filter short/common entities
            if len(entity) <= 2 or entity.lower() in COMMON_WORDS:
                continue

            neighbors = nx_manager.get_neighbors(entity, max_depth=2)
            neighbor_sample = list(neighbors)[:10]  # ✅ Cap neighbors per entity

            for neighbor in neighbor_sample:
                if neighbor == entity:
                    continue

                neighbor_sent_ids = set()
                for triplet in triplets:
                    if triplet['subject'] == neighbor or triplet['object'] == neighbor:
                        neighbor_sent_ids.add(triplet['unique_sentence_id'])

                same_sentence = sent_id in neighbor_sent_ids

                entity_prefix = entity.split("_")[0] if "_" in entity else entity
                neighbor_prefix = neighbor.split("_")[0] if "_" in neighbor else neighbor

                # Soft label logic
                if same_sentence:
                    label = 1.0
                elif entity_prefix == neighbor_prefix:
                    label = 0.3
                else:
                    label = 0.0

                boundary_data.append({
                    'source': entity,
                    'target': neighbor,
                    'label': label,
                    'source_token': entity_prefix,
                    'target_token': neighbor_prefix
                })

    print(f"✅ Created {len(boundary_data)} boundary training pairs.")
    return boundary_data



def create_knowledge_graph_if_needed():
    """Create knowledge graph and boundary data if not already done"""
    if not os.path.exists('knowledge_graph_data.pkl'):
        print("Creating NetworkX knowledge graph and boundary data...")
        with open('svo_triplets.pkl', 'rb') as f:
            data = pickle.load(f)
        train_triplets = data['train_triplets']
        test_triplets = data['test_triplets']

        nx_manager = NetworkXGraphManager()
        nx_manager.create_graph_from_triplets(train_triplets + test_triplets)
        nx_manager.save_graph('knowledge_graph.pkl')

        train_boundary_data = create_boundary_training_data(train_triplets + test_triplets, data['train_sentence_map'], nx_manager)
        test_boundary_data = create_boundary_training_data(train_triplets + test_triplets, data['test_sentence_map'], nx_manager)

        graph_data = {
            'nx_graph': nx_manager,
            'train_boundary_data': train_boundary_data,
            'test_boundary_data': test_boundary_data,
            'sentence_entities': nx_manager.sentence_entities
        }
        with open('knowledge_graph_data.pkl', 'wb') as f:
            pickle.dump(graph_data, f)
        print("Knowledge graph creation complete! Data saved to 'knowledge_graph_data.pkl'")
    else:
        print("Knowledge graph data found. Skipping creation.")

# --- Step 5: Node2Vec Embeddings and Feature Extraction ---
from node2vec import Node2Vec
from gensim.models import Word2Vec

def calculate_graph_features(graph, source_entity, target_entity):
    """Calculate graph-based features between two entities"""
    try:
        if nx.has_path(graph, source_entity, target_entity):
            path_length = nx.shortest_path_length(graph, source_entity, target_entity)
        else:
            path_length = 10

        source_neighbors = set(graph.neighbors(source_entity))
        target_neighbors = set(graph.neighbors(target_entity))
        common_neighbors = len(source_neighbors.intersection(target_neighbors))

        union_neighbors = len(source_neighbors.union(target_neighbors))
        jaccard_sim = common_neighbors / union_neighbors if union_neighbors > 0 else 0

        source_degree = graph.degree(source_entity)
        target_degree = graph.degree(target_entity)

        undirected_graph = graph.to_undirected()
        source_clustering = nx.clustering(undirected_graph, source_entity)
        target_clustering = nx.clustering(undirected_graph, target_entity)

        return {
            'path_length': path_length,
            'common_neighbors': common_neighbors,
            'jaccard_similarity': jaccard_sim,
            'source_degree': source_degree,
            'target_degree': target_degree,
            'source_clustering': source_clustering,
            'target_clustering': target_clustering,
            'degree_diff': abs(source_degree - target_degree)
        }
    except Exception as e:
        # print(f"Error in calculate_graph_features: {e}") # Uncomment for detailed debugging
        return {
            'path_length': 10, 'common_neighbors': 0, 'jaccard_similarity': 0,
            'source_degree': 0, 'target_degree': 0, 'source_clustering': 0,
            'target_clustering': 0, 'degree_diff': 0
        }

def calculate_embedding_features(node_embeddings, source_entity, target_entity):
    """Calculate embedding-based features"""
    source_emb = node_embeddings.get(source_entity, np.zeros(128))
    target_emb = node_embeddings.get(target_entity, np.zeros(128))

    cos_sim = np.dot(source_emb, target_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8)
    euclidean_dist = np.linalg.norm(source_emb - target_emb)

    diff = source_emb - target_emb
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    diff_max = np.max(np.abs(diff))

    return {
        'cosine_similarity': cos_sim,
        'euclidean_distance': euclidean_dist,
        'embedding_diff_mean': diff_mean,
        'embedding_diff_std': diff_std,
        'embedding_diff_max': diff_max
    }
    # print("  Calculated embedding features.") # Uncomment for detailed debugging

from tqdm import tqdm
import pandas as pd
import numpy as np

def create_feature_matrix(boundary_data, graph, node_embeddings):
    """Create feature matrix from boundary data with error handling and progress bar"""
    features = []
    labels = []

    print(f"🧩 Creating feature matrix for {len(boundary_data)} pairs...")

    for item in tqdm(boundary_data, desc="🔄 Generating features"):
        try:
            source = item['source']
            target = item['target']
            label = item['label']

            graph_features = calculate_graph_features(graph, source, target)
            embedding_features = calculate_embedding_features(node_embeddings, source, target)
            feature_vector = {**graph_features, **embedding_features}

            features.append(feature_vector)
            labels.append(label)
        except Exception as e:
            print(f"⚠️ Skipping pair ({item.get('source')}, {item.get('target')}): {e}")
            continue

    df = pd.DataFrame(features)

    if df.empty:
        print("❌ ERROR: No features were generated. Please inspect boundary_data or graph inputs.")
    else:
        print(f"✅ Feature matrix generated with shape: {df.shape}")

    return df, np.array(labels)


def generate_embeddings_and_features_if_needed():
    """Generate Node2Vec embeddings and features if not already done"""
    if not os.path.exists('embeddings_and_features.pkl'):
        print("🧠 Generating Node2Vec embeddings and features...")

        # Load graph and boundary data
        with open('knowledge_graph_data.pkl', 'rb') as f:
            graph_data = pickle.load(f)
        nx_manager = graph_data['nx_graph']
        train_boundary_data = graph_data['train_boundary_data']
        test_boundary_data = graph_data['test_boundary_data']
        graph = nx_manager.graph

        print("🔄 Creating Node2Vec model...")
        undirected_graph = graph.to_undirected()
        node2vec = Node2Vec(undirected_graph, dimensions=128, walk_length=40, num_walks=5, workers=2, p=1, q=1)

        print("🔁 Starting Word2Vec training...")
        model = node2vec.fit(window=10, min_count=1, batch_words=4, epochs=5)
        print("✅ Word2Vec training done.")

        # Generate node embeddings
        node_embeddings = {}
        for node in graph.nodes():
            try:
                node_embeddings[node] = model.wv[node]
            except KeyError:
                node_embeddings[node] = np.zeros(128)

        print("🧩 Generating feature matrices...")

        # Feature matrix creation
        train_features, train_labels = create_feature_matrix(train_boundary_data, graph, node_embeddings)
        print(f"✅ Created training feature matrix with {len(train_features)} rows.")

        test_features, test_labels = create_feature_matrix(test_boundary_data, graph, node_embeddings)
        print(f"✅ Created test feature matrix with {len(test_features)} rows.")

        # Sanitize and scale
        train_features = train_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)

        # Simulated tokens and entropy scores for CRF
        train_tokens = [entry.get('source_token', 'UNK') for entry in train_boundary_data]
        train_entropy_scores = [uniform(0.3, 0.7) for _ in train_tokens]  # Placeholder for now


        # ✅ Save all required components
        embeddings_data = {
            'node_embeddings': node_embeddings,
            'train_features': train_features,
            'test_features': test_features,
            'train_features_scaled': train_features_scaled,
            'test_features_scaled': test_features_scaled,
            'train_labels': train_labels,
            'test_labels': test_labels,
            'scaler': scaler,
            'feature_names': list(train_features.columns),
            'node2vec_model': model,
            'train_tokens': train_tokens,
            'train_entropy_scores': train_entropy_scores  # Used by CRF
        }

        with open('embeddings_and_features.pkl', 'wb') as f:
            pickle.dump(embeddings_data, f)

        print("✅ Embeddings and features saved to 'embeddings_and_features.pkl'")

        # ✅ Trigger model training here if desired
        train_model_if_needed()
    else:
        print("✅ Embeddings and features found. Skipping generation.")

# --- Step 6: Entropy-Based Boundary Detection Model ---
class EntropyBoundaryDetector(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(EntropyBoundaryDetector, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # No Sigmoid!
        self.network = nn.Sequential(*layers)

        self.entropy_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        boundary_logits = self.network(x)           # No sigmoid here
        entropy_score = self.entropy_layer(x)       # This can remain as is
        return boundary_logits.squeeze(), entropy_score.squeeze()

def train_neural_network(X_train, y_train, X_val, y_val, epochs=20, batch_size=128):
    """Train the entropy-based neural network with BCE, entropy loss, KL divergence, and temperature scaling"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = EntropyBoundaryDetector(input_dim=X_train.shape[1])
    model.to(device)

    from torch.nn import BCEWithLogitsLoss
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            boundary_logits, entropy_score = model(batch_X)
            boundary_loss = criterion(boundary_logits, batch_y)
            entropy_loss = -torch.mean(entropy_score * batch_y)

            # KL-Divergence Loss
            probs = torch.softmax(boundary_logits, dim=0)
            target_dist = batch_y / (torch.sum(batch_y) + 1e-8)
            kl_div = torch.nn.functional.kl_div(
                torch.log(probs + 1e-6), target_dist, reduction='batchmean'
            )

            total_loss = boundary_loss + 0.1 * entropy_loss + 0.05 * kl_div
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_device = X_val_tensor.to(device)
            y_val_device = y_val_tensor.to(device)

            val_logits, val_entropy_score = model(X_val_device)
            val_boundary_prob = torch.sigmoid(val_logits)
            val_boundary_loss = criterion(val_logits, y_val_device)
            val_entropy_loss = -torch.mean(val_entropy_score * y_val_device)
            val_total_loss = val_boundary_loss + 0.1 * val_entropy_loss

        # Dynamic threshold tuning
        from sklearn.metrics import f1_score
        val_probs = val_boundary_prob.cpu().numpy().flatten()
        y_val_true = (y_val_tensor.cpu().numpy() >= 0.5).astype(int)  # 🔁 conver

        best_f1 = 0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (val_probs >= t).astype(int)
            f1 = f1_score(y_val_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        print(f"✅ Best threshold: {best_thresh:.2f} with F1: {best_f1:.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        val_loss = val_total_loss.item()
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if epoch % 1 == 0:  # Print every epoch; use %10 for every 10
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Best F1: {best_f1:.4f} | "
                  f"Best Thresh: {best_thresh:.2f} | KL: {kl_div.item():.4f}")
    

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f} | KL: {kl_div.item():.4f}")

    model.load_state_dict(best_model_state)

    # ✅ Temperature Scaling Calibration
    best_temp = 1.0
    best_ece = float('inf')
    temps = np.arange(0.5, 3.0, 0.1)
    for T in temps:
        scaled_probs = torch.sigmoid(val_logits / T).cpu().numpy().flatten()
        ece = expected_calibration_error(scaled_probs, y_val_true)
        if ece < best_ece:
            best_temp = T
            best_ece = ece

    print(f"✅ Best temperature: {best_temp:.2f} with ECE: {best_ece:.4f}")

    # ✅ Store best threshold and temperature in model (attach as attributes)
    model.best_threshold = best_thresh
    model.temperature = best_temp
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Final evaluation on validation set using best threshold
    final_preds = (val_probs >= best_thresh).astype(int)

    final_precision = precision_score(y_val_true, final_preds)
    final_recall = recall_score(y_val_true, final_preds)
    final_f1 = f1_score(y_val_true, final_preds)

    print("\n📊 Final Validation Metrics:")
    print(f"   Precision: {final_precision:.4f}")
    print(f"   Recall   : {final_recall:.4f}")
    print(f"   F1 Score : {final_f1:.4f}")


    return model

def train_model_if_needed():
    """Train and save the neural network and CRF model if not already saved"""
    if not os.path.exists('trained_models.pkl'):
        print("Training neural network model...")

        # Load graph features
        with open('embeddings_and_features.pkl', 'rb') as f:
            data = pickle.load(f)

        train_features_scaled = data['train_features_scaled']
        train_labels = data['train_labels']
        scaler = data['scaler']
        feature_names = data['feature_names']
        original_tokens = data['train_tokens']  # ✅ This must be saved during feature generation
        entropy_scores = data['train_entropy_scores']  # ✅ Save these in your feature step

        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            train_features_scaled, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )

        # ✅ Train entropy-based neural model
        nn_model = train_neural_network(X_train_split, y_train_split, X_val_split, y_val_split, epochs=100)

        # ✅ Load best threshold from training (you MUST return it from train_neural_network)
        best_thresh = nn_model.best_threshold if hasattr(nn_model, 'best_threshold') else 0.7

        # ✅ Train CRF model from entropy predictions
        import sklearn_crfsuite
        from crf_utils import prepare_crf_data  # 🔁 You'll create this util

        X_crf, y_crf = prepare_crf_data(original_tokens, entropy_scores, threshold=best_thresh)

        crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf_model.fit(X_crf, y_crf)

        # ✅ Save both models
        models_data = {
            'neural_network': nn_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'best_threshold': best_thresh,
            'temperature': nn_model.temperature  # ✅ Add this
        }
        with open('trained_models.pkl', 'wb') as f:
            pickle.dump(models_data, f)

        with open('crf_model.pkl', 'wb') as f:
            pickle.dump(crf_model, f)

        print("✅ Model training complete! Saved to 'trained_models.pkl' and 'crf_model.pkl'")
    else:
        print("✅ Trained model found. Skipping training.")

# --- Step 8: Graph Traversal with Your Entropy Model ---
class GraphTraversalWithEntropyModel:
    def __init__(self, graph=None, model=None, scaler=None, node_embeddings=None, threshold=0.7, temperature=1.0):
        self.graph = graph
        self.model = model
        self.scaler = scaler
        self.node_embeddings = node_embeddings
        self.threshold = threshold  # ← from loaded model if available
        self.temperature = temperature

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model:
            self.model.to(self.device)
            self.model.eval()

        # Removed self.create_sentence_mapping() as it's for static data
    def extract_syntactic_features(self, token):
        token_lower = token.lower()
        return np.array([
            int(token[0].isupper()) if token else 0,               # Is capitalized
            int(token in ['.', '!', '?']),                         # Is punctuation
            int(token.endswith('.')),                              # Ends with period
            int(token_lower in ['he', 'she', 'the', 'this', 'that']),  # Common sentence starters
            int(len(token) <= 2),                                  # Very short token (e.g., 'Dr', 'a')
        ], dtype=np.float32)


    def calculate_features(self, source_token, target_token):
        try:
            direct_connection = 1.0 if self.graph.has_edge(source_token, target_token) else 0.0
            path_length = nx.shortest_path_length(self.graph, source_token, target_token) if nx.has_path(self.graph, source_token, target_token) else 10.0

            source_neighbors = set(self.graph.neighbors(source_token))
            target_neighbors = set(self.graph.neighbors(target_token))
            common_neighbors = len(source_neighbors.intersection(target_neighbors))
            jaccard_similarity = (common_neighbors /
                                len(source_neighbors.union(target_neighbors))
                                if source_neighbors.union(target_neighbors) else 0)

            source_degree = self.graph.degree(source_token)
            target_degree = self.graph.degree(target_token)
            degree_difference = abs(source_degree - target_degree)

            if source_token in self.node_embeddings and target_token in self.node_embeddings:
                source_emb = self.node_embeddings[source_token]
                target_emb = self.node_embeddings[target_token]
                emb_similarity = cosine_similarity([source_emb], [target_emb])[0][0]
                emb_distance = np.linalg.norm(source_emb - target_emb)
                emb_distance_norm = emb_distance / (np.linalg.norm(source_emb) + np.linalg.norm(target_emb) + 1e-8)
            else:
                emb_similarity = 0.5
                emb_distance_norm = 0.5

            # Graph and embedding features
            graph_embed_features = np.array([
                direct_connection,
                path_length / 10.0,
                jaccard_similarity,
                source_degree / 10.0,
                target_degree / 10.0,
                degree_difference / 10.0,
                common_neighbors / 5.0,
                emb_similarity,
                emb_distance_norm,
                len(source_neighbors) / 10.0
            ], dtype=np.float32)

            # NEW: Syntactic features on the source token
            syntactic_features = self.extract_syntactic_features(source_token)

            # Combine all features
            combined_features = np.concatenate([graph_embed_features, syntactic_features])
            return combined_features

        except Exception as e:
            # Fallback for tokens not found in the graph
            # This might happen for new words in user input not seen during training
            return np.array([0.5] * 15) 

    def position_based_entropy(self, token):
        token_lower = token.lower()

        # High-entropy tokens that should NOT trigger a boundary themselves
        non_boundary_punct = ["p.m.", "a.m.", "mr.", "dr.", "mrs.", "etc."]

        if token_lower in non_boundary_punct:
            return 0.85 + np.random.normal(0, 0.05)  # High entropy, but ignore as boundary marker

        if token in [".", "!", "?"]:
            return 0.90 + np.random.normal(0, 0.03)

        if token in ["He", "She", "The", "This", "That"] or token[0].isupper():
            return 0.80 + np.random.normal(0, 0.10)

        if token_lower in ["the", "a", "an", "and", "or", "but", "at", "in", "on"]:
            return 0.15 + np.random.normal(0, 0.05)

        return 0.35 + np.random.normal(0, 0.10)


    def calculate_entropy_score(self, current_token, context_tokens):
        if not context_tokens or self.model is None or self.scaler is None or self.graph is None or self.node_embeddings is None:
            return self.position_based_entropy(current_token)

        feature_vectors = []
        for context_token in context_tokens:
            if context_token != current_token and context_token in self.graph.nodes():
                features = self.calculate_features(current_token, context_token)
                feature_vectors.append(features)

        if not feature_vectors:
            return self.position_based_entropy(current_token)

        try:
            feature_matrix = np.array(feature_vectors)
            feature_matrix_scaled = self.scaler.transform(feature_matrix)

            with torch.no_grad():
                feature_tensor = torch.FloatTensor(feature_matrix_scaled).to(self.device)
                boundary_logits, _ = self.model(feature_tensor)
                scaled_logits = boundary_logits / self.temperature
                boundary_probs = torch.sigmoid(scaled_logits).cpu().numpy().flatten()
            # --- Core scoring components ---
            avg_boundary_prob = np.mean(boundary_probs)
            prob_variance = np.var(boundary_probs)
            entropy_score = avg_boundary_prob + 0.2 * prob_variance
            position_entropy = self.position_based_entropy(current_token)

            # ✅ New: Context entropy from nearby tokens
            context_entropy = np.mean([
                self.position_based_entropy(tok)
                for tok in context_tokens
                if tok != current_token
            ])

            # ✅ Smoothed final entropy score
            final_entropy = (
                0.6 * entropy_score +
                0.2 * context_entropy +
                0.2 * position_entropy
            )
            return min(final_entropy, 1.0)

        except Exception as e:
            return self.position_based_entropy(current_token)

    def detect_sentence_boundaries(self, test_tokens):
            results = []
            print("Calculating entropy scores for each token...")
            print("-" * 60)

            sentence_counter = 0  # Start sentence count
            for i, token in enumerate(test_tokens):
                context_start = max(0, i - 3)
                context_end = min(len(test_tokens), i + 4)
                context_tokens = test_tokens[context_start:context_end]

                entropy_score = self.calculate_entropy_score(token, context_tokens)
                sentence_start_label = ""

                # Use dynamic threshold from self.threshold (set via constructor)
                if entropy_score > self.threshold:
                    prev_token = test_tokens[i - 1].lower() if i > 0 else ""
                    sentence_end_markers = [".", "!", "?", "p.m.", "a.m.", "mr.", "dr.", "mrs.", "etc."]

                    # Trigger new sentence only if preceded by punctuation or if it's the first token
                    if i == 0 or prev_token in sentence_end_markers:
                        sentence_counter += 1
                        sentence_start_label = f"✓ (Sentence {sentence_counter})"

                results.append({
                    'Token': token,
                    'Entropy Score': f"{entropy_score:.2f}",
                    'Sentence Start': sentence_start_label
                })

                print(f"Token: {token:<15} | Entropy: {entropy_score:.2f} | {sentence_start_label}")

            return results



def create_traversal_visualization(results_df, tokens):
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    entropy_values = [float(score) for score in results_df['Entropy Score']]
    boundary_positions = [i for i, start in enumerate(results_df['Sentence Start']) if start]

    ax1.plot(range(len(entropy_values)), entropy_values, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Boundary Threshold (0.7)')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Entropy Score')
    ax1.set_title('Entropy Scores Across Tokens')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for i, entropy in enumerate(entropy_values):
        if entropy > 0.7:
            ax1.annotate(f'{tokens[i]}', (i, entropy), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=8, color='red')

    ax2.scatter(boundary_positions, [entropy_values[i] for i in boundary_positions],
               color='red', s=100, label='Detected Boundaries', zorder=5)
    ax2.plot(range(len(entropy_values)), entropy_values, 'b-', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Entropy Score')
    ax2.set_title('Detected Sentence Boundaries')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.hist(entropy_values, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax3.set_xlabel('Entropy Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Entropy Scores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.barh(range(len(tokens)), entropy_values, color=['red' if e > 0.7 else 'blue' for e in entropy_values])
    ax4.set_yticks(range(len(tokens)))
    ax4.set_yticklabels(tokens)
    ax4.set_xlabel('Entropy Score')
    ax4.set_title('Entropy Score by Token')
    ax4.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Graph Traversal Results with Entropy Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

import sklearn_crfsuite

def prepare_crf_data(tokens, entropy_scores, threshold=0.7):
    X_seq = []
    y_seq = []

    for i, token in enumerate(tokens):
        features = {
            'token': token,
            'entropy': entropy_scores[i],
            'capitalized': token[0].isupper(),
            'ends_with_dot': token.endswith('.'),
            'prev_token': tokens[i - 1] if i > 0 else 'START',
            'next_token': tokens[i + 1] if i < len(tokens) - 1 else 'END',
        }
        X_seq.append(features)
        y_seq.append('B' if entropy_scores[i] > threshold else 'O')

    return [X_seq], [y_seq]

def run_inference():
    print("=" * 80)
    print("GRAPH TRAVERSAL WITH YOUR ENTROPY MODEL - INFERENCE")
    print("=" * 80)

    print("\n1. Loading trained model and data for inference...")
    try:
        with open('knowledge_graph_data.pkl', 'rb') as f:
            graph_data = pickle.load(f)
        with open('trained_models.pkl', 'rb') as f:
            models_data = pickle.load(f)
        with open('embeddings_and_features.pkl', 'rb') as f:
            features_data = pickle.load(f)
        with open('crf_model.pkl', 'rb') as f:
            crf_model = pickle.load(f)

        graph = graph_data['nx_graph'].graph
        model = models_data['neural_network']
        scaler = models_data['scaler']
        node_embeddings = features_data['node_embeddings']
        threshold = models_data.get('best_threshold', 0.7)
        temperature = models_data.get('temperature', 1.0)

        traversal_system = GraphTraversalWithEntropyModel(
            graph=graph,
            model=model,
            scaler=scaler,
            node_embeddings=node_embeddings,
            threshold=threshold,
            temperature=temperature
        )
        print("   Model and data loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Required data files not found. Please run training first.")
        return

    print("\n2. Enter a sentence to analyze (or press Enter to use a default test sentence):")
    user_input = input(">> ").strip()
    if not user_input:
        user_input = "Dr. Smith arrived at 5 p.m. He began the experiment. The results were remarkable."

    ttokens = word_tokenize(user_input)
    tokens = [t.strip().lower() for t in ttokens if t.strip()]
    print("\n🧪 Normalized Tokens for Entropy:")
    print(tokens)

    print("\n3. Detecting sentence boundaries using entropy model...")
    results = traversal_system.detect_sentence_boundaries(tokens)

    from crf_utils import prepare_crf_data
    crf_input_X, _ = prepare_crf_data(
        tokens=[row['Token'] for row in results],
        entropy_scores=[float(row['Entropy Score']) for row in results],
        threshold=threshold
    )
    crf_labels = crf_model.predict(crf_input_X)[0]

    sentence_id = 1
    for i, row in enumerate(results):
        if crf_labels[i] == 'B':
            row['Sentence Start'] = f"✓ (Sentence {sentence_id})"
            sentence_id += 1
        else:
            row['Sentence Start'] = ""

    print("\n4. Results:")
    print("-" * 60)
    for row in results:
        print(f"Token: {row['Token']:<15} | Entropy: {row['Entropy Score']} | {row['Sentence Start']}")

    user_input_sentences = input("\nEnter multiple sentences (e.g., 'Dr. Smith arrived at 5 p.m. He immediately began the experiment. The results surprised everyone.'): ")
    test_tokens = word_tokenize(user_input_sentences)
    test_tokens = [t.strip().lower() for t in test_tokens if t.strip()]
    print("\n🧪 Normalized Tokens for Final Inference:")
    print(test_tokens)

    print(f"\n2. Processing input with {len(test_tokens)} tokens...")
    print(f"   Input sentence: {user_input_sentences}")

    print(f"\n3. Running entropy-based boundary detection...")
    results = traversal_system.detect_sentence_boundaries(test_tokens)

    crf_input_X, _ = prepare_crf_data(
        tokens=[row['Token'] for row in results],
        entropy_scores=[float(row['Entropy Score']) for row in results],
        threshold=threshold
    )
    crf_labels = crf_model.predict(crf_input_X)[0]

    sentence_id = 1
    for i, row in enumerate(results):
        if crf_labels[i] == 'B':
            row['Sentence Start'] = f"✓ (Sentence {sentence_id})"
            sentence_id += 1
        else:
            row['Sentence Start'] = ""

    if not results or not isinstance(results, list):
        print("❌ Inference failed or returned empty results.")
        return

    print(f"\n📦 Raw inference results (first 3 rows): {results[:3]}")
    results_df = pd.DataFrame(results)

    print(f"\n4. RESULTS TABLE (Your Format):")
    print("=" * 60)
    print(f"{'Token':<15} | {'Entropy Score':<12} | {'Sentence Start'}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['Token']:<15} | {row['Entropy Score']:<12} | {row['Sentence Start']}")
    print("=" * 60)

    print("\n📋 Columns in results_df:", results_df.columns.tolist())
    print(results_df.head())

    if 'Entropy Score' not in results_df.columns:
        print("❌ Column 'Entropy Score' missing in results_df. Found columns:", results_df.columns.tolist())
        return

    entropy_values = results_df['Entropy Score'].astype(float).tolist()
    boundaries_detected = sum(1 for start in results_df['Sentence Start'] if start)

    print(f"\n5. ANALYSIS:")
    print(f"   - Boundaries detected: {boundaries_detected}")
    print(f"   - Average entropy: {np.mean(entropy_values):.3f}")
    print(f"   - Max entropy: {max(entropy_values):.3f}")
    print(f"   - Tokens above threshold (0.7): {sum(1 for e in entropy_values if e > 0.7)}")

    print(f"\n6. Creating visualization...")
    create_traversal_visualization(results_df, test_tokens)

    from sklearn.metrics import precision_score, recall_score, f1_score
    test_features_scaled = features_data['test_features_scaled']
    test_labels = np.array(features_data['test_labels']).astype(int)

    model.eval()
    with torch.no_grad():
        test_logits, _ = model(torch.FloatTensor(test_features_scaled))
        test_probs = torch.sigmoid(test_logits).numpy().flatten()

    test_preds = (test_probs >= threshold).astype(int)

    precision = precision_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)

    print("\n📊 FINAL EVALUATION ON TEST DATA")
    print("-" * 60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\n" + '=' * 80)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print('=' * 80)


if __name__ == "__main__":
    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except nltk.downloader.DownloadError:
        nltk.download('punkt_tab')

    # Ensure spaCy model is downloaded
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        os.system("python -m spacy download en_core_web_sm")
        print("SpaCy model downloaded.")

    # Run all data preparation and training steps if the final model file doesn't exist
    if not os.path.exists('trained_models.pkl'):
        print("\n--- Running full training pipeline as model is not found ---")
        preprocess_data_if_needed()
        split_data_if_needed()
        extract_svo_if_needed()
        create_knowledge_graph_if_needed()
        generate_embeddings_and_features_if_needed()
        train_model_if_needed()
    else:
        print("\n--- Trained model found. Skipping training pipeline. ---")

    # Always run inference after ensuring the model is available
    run_inference()
