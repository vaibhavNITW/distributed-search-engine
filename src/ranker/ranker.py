
import os
import pickle
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

print("🔥 Ranker script is starting...") 

# Configure logging
logging.basicConfig(
    filename='logs/ranker.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MLRanker:
    def __init__(self):
        """Initialize the ML-based ranker"""
        self.model = None
        self.feature_extractor = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('data/models', exist_ok=True)
    
    def extract_features(self, query, document):
        """
        Extract features from query-document pair
        
        Args:
            query (str): Search query
            document (dict): Document data
            
        Returns:
            dict: Feature vector
        """
        # Extract query and document title and text
        query_text = query.lower()
        title = document.get('title', '').lower()
        
        # Simple features
        features = {
            'title_query_overlap': self._count_term_overlap(query_text, title),
            'title_length': len(title.split()),
            'query_length': len(query_text.split()),
        }
        
        return features
    
    def _count_term_overlap(self, text1, text2):
        """Count the number of overlapping terms between two texts"""
        terms1 = set(text1.lower().split())
        terms2 = set(text2.lower().split())
        return len(terms1.intersection(terms2))
    
    def prepare_training_data(self, search_logs):
        """
        Prepare training data from search logs
        
        Args:
            search_logs (list): List of search log entries
                Each entry should be a dict with:
                - query: search query
                - results: list of result documents
                - clicks: list of clicked document indices
            
        Returns:
            tuple: X (features), y (labels)
        """
        X = []
        y = []
        
        for log in search_logs:
            query = log['query']
            results = log['results']
            clicks = log['clicks']
            
            for i, result in enumerate(results):
                # Create feature vector
                features = self.extract_features(query, result)
                
                # Convert dict to list in a consistent order
                feature_vector = [features[k] for k in sorted(features.keys())]
                
                # Add to features
                X.append(feature_vector)
                
                # Label: 1 if clicked, 0 if not
                label = 1 if i in clicks else 0
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, search_logs, model_type='logreg'):
        """
        Train the ranking model
        
        Args:
            search_logs (list): List of search log entries
            model_type (str): Type of model to train ('logreg' or 'svm')
        """
        try:
            # Prepare training data
            X, y = self.prepare_training_data(search_logs)
            
            # Split into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create model pipeline
            if model_type == 'logreg':
                model = LogisticRegression(C=1.0, class_weight='balanced')
            elif model_type == 'svm':
                model = LinearSVC(C=1.0, class_weight='balanced')
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = pipeline.score(X_val, y_val)
            logging.info(f"Validation accuracy: {val_score:.4f}")
            
            # Save as the main model
            self.model = pipeline
            
            logging.info(f"Trained {model_type} model on {len(X)} examples")
            
            # Save model
            self.save_model()
            
            return val_score
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return None
    
    def rank(self, query, documents):
        """
        Rank documents based on relevance to query
        
        Args:
            query (str): Search query
            documents (list): List of document objects
            
        Returns:
            list: Ranked documents with scores
        """
        if not self.model or not documents:
            return documents
        
        try:
            # Extract features for each document
            features = []
            for document in documents:
                doc_features = self.extract_features(query, document)
                # Convert dict to list in a consistent order
                feature_vector = [doc_features[k] for k in sorted(doc_features.keys())]
                features.append(feature_vector)
            
            # Convert to numpy array
            features = np.array(features)
            
            # Get relevance scores
            if hasattr(self.model, 'predict_proba'):
                # For models that support probability estimates
                scores = self.model.predict_proba(features)[:, 1]  # Probability of class 1
            else:
                # For models that only support decision function
                scores = self.model.decision_function(features)
            
            # Combine with original documents
            scored_docs = list(zip(documents, scores))
            
            # Sort by score (descending)
            ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
            # Return documents with scores
            return [{'document': doc, 'ml_score': float(score)} for doc, score in ranked_docs]
            
        except Exception as e:
            logging.error(f"Error ranking documents: {str(e)}")
            return [{'document': doc, 'ml_score': 0.0} for doc in documents]
    
    def save_model(self, filename='data/models/ranker_model.pkl'):
        """Save model to file"""
        if self.model:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            logging.info(f"Model saved to {filename}")
    
    def load_model(self, filename='data/models/ranker_model.pkl'):
        """Load model from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)
            logging.info(f"Model loaded from {filename}")
            return True
        else:
            logging.warning(f"Model file not found: {filename}")
            return False

    def generate_sample_training_data(self, num_samples=100):
        """
        Generate synthetic training data for demonstration purposes
        
        Args:
            num_samples (int): Number of search log samples to generate
            
        Returns:
            list: Synthetic search logs
        """
        # Sample queries
        queries = [
            "python programming",
            "machine learning",
            "data structures",
            "web development",
            "artificial intelligence",
            "computer science",
            "software engineering",
            "database management",
            "network security",
            "cloud computing"
        ]
        
        # Generate synthetic logs
        logs = []
        
        for _ in range(num_samples):
            # Select random query
            query = np.random.choice(queries)
            
            # Generate random results (2-10 results per query)
            num_results = np.random.randint(2, 11)
            results = []
            
            for i in range(num_results):
                # Create dummy document
                result = {
                    'title': f"Document about {query.title()} - Part {i+1}",
                    'score': np.random.random(),
                    'url': f"https://example.com/doc{i+1}"
                }
                results.append(result)
            
            # Simulate clicks (prefer documents with "query" terms in title)
            clicks = []
            for i, result in enumerate(results):
                # Higher chance of clicking if title contains query terms
                click_prob = 0.2  # Base probability
                
                # Boost probability if title contains query terms
                if self._count_term_overlap(query, result['title']) > 0:
                    click_prob += 0.4
                
                # Add position bias (earlier results more likely to be clicked)
                click_prob *= (1.0 - i * 0.1) if i < 5 else 0.1
                
                if np.random.random() < click_prob:
                    clicks.append(i)
            
            # Add to logs
            logs.append({
                'query': query,
                'results': results,
                'clicks': clicks
            })
        
        logging.info(f"Generated {num_samples} synthetic search logs")
        return logs

# Example usage
if __name__ == "__main__":
    ranker = MLRanker()

    # Generate synthetic training data
    search_logs = ranker.generate_sample_training_data(200)

    # Train model
    ranker.train(search_logs, model_type='logreg')

    # Example ranking
    query = "machine learning algorithms"
    documents = [
        {'title': 'Introduction to Machine Learning', 'url': 'https://example.com/1'},
        {'title': 'Advanced Python Programming', 'url': 'https://example.com/2'},
        {'title': 'Machine Learning Algorithms Explained', 'url': 'https://example.com/3'},
        {'title': 'Web Development Basics', 'url': 'https://example.com/4'}
    ]

    ranked_docs = ranker.rank(query, documents)

    print("Ranked Documents:")
    for i, item in enumerate(ranked_docs, 1):
        doc = item['document']
        score = item['ml_score']
        print(f"{i}. {doc['title']} (Score: {score:.4f})")
