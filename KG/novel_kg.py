import spacy
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse
import re
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import json
##python novel_kg.py E:/LLM/LightRAG/KG/RobertLouisStevenson.txt  --visualize
# Set environment variables
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Download necessary NLTK data
nltk.download('punkt', quiet=True)


class KnowledgeGraphBuilder:
    def __init__(self, spacy_model="en_core_web_sm", bert_model='dbmdz/bert-large-cased-finetuned-conll03-english'):
        # Load spaCy model
        print("Loading spaCy model...")
        self.nlp = spacy.load(spacy_model)

        # Load BERT model and tokenizer
        print("Loading BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertForTokenClassification.from_pretrained(bert_model)

        # Create NER pipeline
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        # Initialize knowledge graph
        self.graph = nx.Graph()

        # Track entity mentions for co-reference resolution
        self.entity_mentions = {}

        # Define relationship patterns
        self.relationship_patterns = [
            # Person-Location patterns
            (r"(\w+)\s+(?:lived|resided|stayed)\s+(?:in|at)\s+(.+?)(?:\.|\,)", "lived_in"),
            (r"(\w+)\s+(?:visited|went\s+to|traveled\s+to)\s+(.+?)(?:\.|\,)", "visited"),

            # Person-Person patterns
            (r"(\w+)\s+(?:met|encountered|saw)\s+(\w+)", "met"),
            (r"(\w+)\s+(?:is|was)\s+(?:a\s+)?(?:friend|colleague|associate)\s+of\s+(\w+)", "friend_of"),
            (r"(\w+)\s+(?:is|was)\s+(?:married|engaged|related)\s+to\s+(\w+)", "related_to"),

            # Person-Object patterns
            (r"(\w+)\s+(?:had|owned|possessed|carried)\s+(?:a|an|the)\s+(.+?)(?:\.|\,)", "owned"),
            (r"(\w+)\s+(?:used|utilized|employed)\s+(?:a|an|the)\s+(.+?)(?:\.|\,)", "used"),

            # Action patterns
            (r"(\w+)\s+(?:investigated|examined|studied)\s+(?:a|an|the)\s+(.+?)(?:\.|\,)", "investigated"),
        ]

    def load_text_file(self, file_path):
        """Load text from a file."""
        print(f"Loading text from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def preprocess_text(self, text):
        """Clean and preprocess the text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into sentences
        sentences = sent_tokenize(text)
        return sentences

    def extract_entities_from_sentence(self, sentence):
        """Extract named entities from a sentence using both spaCy and BERT."""
        # Use spaCy for entity extraction
        doc = self.nlp(sentence)
        entities_spacy = [(ent.text, ent.label_) for ent in doc.ents]

        # Use BERT for entity extraction
        try:
            entities_bert = self.ner_pipeline(sentence)
            entities_bert = [(entity['word'], entity['entity']) for entity in entities_bert]
        except Exception as e:
            print(f"Error in BERT processing: {e}")
            entities_bert = []

        # Combine and deduplicate entities
        all_entities = set(entities_spacy + entities_bert)

        # Update entity mentions dictionary for co-reference resolution
        for entity, label in all_entities:
            if label in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'FAC', 'WORK_OF_ART']:
                if entity not in self.entity_mentions:
                    self.entity_mentions[entity] = []
                self.entity_mentions[entity].append((sentence, label))

        return list(all_entities)

    def extract_relations_from_sentence(self, sentence):
        """Extract relationships between entities in a sentence."""
        relations = []

        # Apply all relationship patterns
        for pattern, relation_type in self.relationship_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                object_ = match.group(2).strip()
                relations.append((subject, relation_type, object_))

        # Also extract entities connected by "and" or similar conjunctions
        doc = self.nlp(sentence)
        for token in doc:
            if token.dep_ == "conj" and token.head.pos_ == "PROPN" and token.pos_ == "PROPN":
                relations.append((token.head.text, "associated_with", token.text))

        return relations

    def build_knowledge_graph(self, text):
        """Build a knowledge graph from text."""
        # Preprocess text
        sentences = self.preprocess_text(text)
        print(f"Processing {len(sentences)} sentences...")

        # Process each sentence
        for sentence in tqdm(sentences):
            # Extract entities
            entities = self.extract_entities_from_sentence(sentence)

            # Add entities as nodes
            for entity, label in entities:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, label=label)

            # Extract and add relationships
            relations = self.extract_relations_from_sentence(sentence)
            for relation in relations:
                subject, relation_type, object_ = relation
                if self.graph.has_node(subject) and self.graph.has_node(object_):
                    self.graph.add_edge(subject, object_, relation=relation_type)

        # Handle co-reference resolution by connecting entities with similar mentions
        self._resolve_coreferences()

        print(f"Knowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph

    def _resolve_coreferences(self):
        """Simple co-reference resolution by connecting entities that occur in similar contexts."""
        # This is a simple implementation; a more sophisticated approach would use a dedicated co-reference model
        for entity1, mentions1 in self.entity_mentions.items():
            for entity2, mentions2 in self.entity_mentions.items():
                if entity1 != entity2 and self.graph.has_node(entity1) and self.graph.has_node(entity2):
                    # Check if entities appear in the same sentences
                    common_sentences = set([m[0] for m in mentions1]) & set([m[0] for m in mentions2])
                    if len(common_sentences) > 0:
                        # Add a co-occurrence edge if they appear in the same sentence multiple times
                        if len(common_sentences) >= 2:
                            self.graph.add_edge(entity1, entity2, relation="co_occurs_with",
                                                weight=len(common_sentences))

    def save_knowledge_graph(self, output_dir="output"):
        """Save the knowledge graph to files."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save graph as GraphML
        nx.write_graphml(self.graph, os.path.join(output_dir, "knowledge_graph.graphml"))

        # Save nodes and edges as CSV
        nodes_df = pd.DataFrame([
            {"Node": node, "Label": data.get("label", "UNKNOWN")}
            for node, data in self.graph.nodes(data=True)
        ])
        nodes_df.to_csv(os.path.join(output_dir, "nodes.csv"), index=False)

        edges_df = pd.DataFrame([
            {"Source": u, "Target": v, "Relation": data.get("relation", ""), "Weight": data.get("weight", 1)}
            for u, v, data in self.graph.edges(data=True)
        ])
        edges_df.to_csv(os.path.join(output_dir, "edges.csv"), index=False)

        # Save as JSON for web visualization
        graph_data = {
            "nodes": [{"id": node, "label": data.get("label", "UNKNOWN")}
                      for node, data in self.graph.nodes(data=True)],
            "links": [{"source": u, "target": v, "relation": data.get("relation", ""),
                       "weight": data.get("weight", 1)}
                      for u, v, data in self.graph.edges(data=True)]
        }
        with open(os.path.join(output_dir, "graph.json"), "w") as f:
            json.dump(graph_data, f)

        print(f"Knowledge graph saved to {output_dir}")

    def visualize_graph(self, output_file="knowledge_graph.png"):
        """Visualize the knowledge graph."""
        print("Visualizing knowledge graph...")
        plt.figure(figsize=(12, 12))

        # Set up node colors based on entity type
        node_colors = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            label = node_data.get("label", "UNKNOWN")
            if "PERSON" in label:
                node_colors.append("lightblue")
            elif "LOC" in label or "GPE" in label:
                node_colors.append("lightgreen")
            elif "ORG" in label:
                node_colors.append("orange")
            else:
                node_colors.append("gray")

        # Draw the graph
        pos = nx.spring_layout(self.graph, k=0.5, iterations=100)
        nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, width=1, alpha=0.5)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)

        # Draw edge labels
        edge_labels = {(u, v): data.get("relation", "") for u, v, data in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved as {output_file}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a knowledge graph from a novel text file")
    parser.add_argument("input_file", help="Path to the novel text file")
    parser.add_argument("--output_dir", default="output", help="Directory to save output files")
    parser.add_argument("--spacy_model", default="en_core_web_sm", help="spaCy model to use")
    parser.add_argument("--bert_model", default="dbmdz/bert-large-cased-finetuned-conll03-english",
                        help="BERT model to use for NER")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")

    # Parse arguments
    args = parser.parse_args()

    # Create knowledge graph builder
    kg_builder = KnowledgeGraphBuilder(spacy_model=args.spacy_model, bert_model=args.bert_model)

    # Load text
    text = kg_builder.load_text_file(args.input_file)

    # Build knowledge graph
    graph = kg_builder.build_knowledge_graph(text)

    # Save knowledge graph
    kg_builder.save_knowledge_graph(output_dir=args.output_dir)

    # Visualize if requested
    if args.visualize:
        kg_builder.visualize_graph(output_file=os.path.join(args.output_dir, "knowledge_graph.png"))

    print("Knowledge graph generation complete!")


if __name__ == "__main__":
    main()