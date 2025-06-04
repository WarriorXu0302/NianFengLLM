import streamlit as st
import os
import tempfile
from novel_kg import KnowledgeGraphBuilder
import networkx as nx
import pyvis.network as net
import pandas as pd
import matplotlib.pyplot as plt
import time
##streamlit run novel_kg_app.py

st.set_page_config(page_title="Novel Knowledge Graph Generator", layout="wide")

st.title("Novel Knowledge Graph Generator")
st.write("""
Upload a novel text file to generate a knowledge graph of entities and their relationships.
This tool extracts characters, locations, and other entities, along with their connections.
""")

# Sidebar for configuration options
st.sidebar.header("Configuration")
spacy_model = st.sidebar.selectbox(
    "Select spaCy model",
    ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
    index=0
)

bert_model = st.sidebar.selectbox(
    "Select BERT model",
    ["dbmdz/bert-large-cased-finetuned-conll03-english",
     "dslim/bert-base-NER"],
    index=0
)

chunk_size = st.sidebar.slider(
    "Text chunk size (sentences)",
    min_value=10,
    max_value=100,
    value=50,
    help="Process text in chunks of this many sentences to prevent memory issues"
)

# File uploader
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

if uploaded_file is not None:
    # Display file info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / 1024:.2f} KB"
    }
    st.write(file_details)

    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    # Process the file when the button is clicked
    if st.button("Generate Knowledge Graph"):
        with st.spinner("Processing text file..."):
            # Display progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create knowledge graph builder
            kg_builder = KnowledgeGraphBuilder(spacy_model=spacy_model, bert_model=bert_model)

            # Load text
            text = kg_builder.load_text_file(temp_file_path)
            status_text.text("Text loaded successfully")
            progress_bar.progress(10)

            # Process text in chunks to show progress
            sentences = kg_builder.preprocess_text(text)
            total_chunks = max(1, len(sentences) // chunk_size)

            # Process each chunk
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(sentences))
                chunk_text = " ".join(sentences[start_idx:end_idx])

                status_text.text(f"Processing chunk {i + 1} of {total_chunks}...")

                # Extract entities and relations for this chunk
                for sentence in sentences[start_idx:end_idx]:
                    entities = kg_builder.extract_entities_from_sentence(sentence)
                    for entity, label in entities:
                        if not kg_builder.graph.has_node(entity):
                            kg_builder.graph.add_node(entity, label=label)

                    relations = kg_builder.extract_relations_from_sentence(sentence)
                    for relation in relations:
                        subject, relation_type, object_ = relation
                        if kg_builder.graph.has_node(subject) and kg_builder.graph.has_node(object_):
                            kg_builder.graph.add_edge(subject, object_, relation=relation_type)

                # Update progress
                progress_bar.progress(10 + 80 * (i + 1) // total_chunks)

            # Resolve coreferences
            status_text.text("Resolving coreferences...")
            kg_builder._resolve_coreferences()
            progress_bar.progress(95)

            # Create output directory
            output_dir = os.path.join(tempfile.gettempdir(), "novel_kg_output")
            os.makedirs(output_dir, exist_ok=True)

            # Save knowledge graph
            status_text.text("Saving knowledge graph...")
            kg_builder.save_knowledge_graph(output_dir=output_dir)

            # Visualize graph
            status_text.text("Creating visualization...")
            kg_builder.visualize_graph(output_file=os.path.join(output_dir, "knowledge_graph.png"))
            progress_bar.progress(100)
            status_text.text("Knowledge graph generation complete!")

            # Display results
            graph = kg_builder.graph

            # Display statistics
            st.subheader("Knowledge Graph Statistics")
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Number of Entities", len(graph.nodes()))
            with stats_col2:
                st.metric("Number of Relationships", len(graph.edges()))

            # Display entity types
            entity_types = {}
            for node, data in graph.nodes(data=True):
                label = data.get("label", "UNKNOWN")
                if label in entity_types:
                    entity_types[label] += 1
                else:
                    entity_types[label] = 1

            # Create DataFrame for entity types
            entity_df = pd.DataFrame({
                "Entity Type": entity_types.keys(),
                "Count": entity_types.values()
            }).sort_values("Count", ascending=False)

            st.subheader("Entity Types")
            st.bar_chart(entity_df, x="Entity Type", y="Count")

            # Display relationship types
            relation_types = {}
            for u, v, data in graph.edges(data=True):
                relation = data.get("relation", "UNKNOWN")
                if relation in relation_types:
                    relation_types[relation] += 1
                else:
                    relation_types[relation] = 1

            # Create DataFrame for relationship types
            relation_df = pd.DataFrame({
                "Relationship Type": relation_types.keys(),
                "Count": relation_types.values()
            }).sort_values("Count", ascending=False)

            st.subheader("Relationship Types")
            st.bar_chart(relation_df, x="Relationship Type", y="Count")

            # Create interactive visualization using pyvis
            st.subheader("Interactive Knowledge Graph")

            # Create network
            network = net.Network(height="600px", width="100%", notebook=True)

            # Add nodes
            for node, data in graph.nodes(data=True):
                label = data.get("label", "UNKNOWN")
                # Choose color based on entity type
                if "PERSON" in label:
                    color = "#66ccff"  # Light blue for people
                elif "LOC" in label or "GPE" in label:
                    color = "#90ee90"  # Light green for locations
                elif "ORG" in label:
                    color = "#ffcc66"  # Orange for organizations
                else:
                    color = "#cccccc"  # Gray for other entities

                network.add_node(node, label=node, title=f"Type: {label}", color=color)

            # Add edges
            for u, v, data in graph.edges(data=True):
                relation = data.get("relation", "")
                weight = data.get("weight", 1)
                network.add_edge(u, v, title=relation, width=weight)

            # Set options
            network.set_options("""
            const options = {
                "nodes": {
                    "shape": "dot",
                    "size": 20,
                    "font": {
                        "size": 14,
                        "face": "Tahoma"
                    }
                },
                "edges": {
                    "color": {
                        "inherit": true
                    },
                    "smooth": false
                },
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.5,
                    "stabilization": {
                        "enabled": true,
                        "iterations": 1000
                    }
                }
            }
            """)

            # Save and display the network
            vis_path = os.path.join(output_dir, "knowledge_graph.html")
            network.save_graph(vis_path)

            with open(vis_path, "r", encoding="utf-8") as f:
                html_string = f.read()

            st.components.v1.html(html_string, height=600)

            # Display tabular data
            st.subheader("Entities")
            nodes_df = pd.DataFrame([
                {"Entity": node, "Type": data.get("label", "UNKNOWN")}
                for node, data in graph.nodes(data=True)
            ])
            st.dataframe(nodes_df)

            st.subheader("Relationships")
            edges_df = pd.DataFrame([
                {"Source": u, "Target": v, "Relation": data.get("relation", ""), "Weight": data.get("weight", 1)}
                for u, v, data in graph.edges(data=True)
            ])
            st.dataframe(edges_df)

            # Option to download results
            st.subheader("Download Results")
            st.download_button(
                label="Download Nodes CSV",
                data=nodes_df.to_csv(index=False),
                file_name="novel_entities.csv",
                mime="text/csv"
            )

            st.download_button(
                label="Download Relationships CSV",
                data=edges_df.to_csv(index=False),
                file_name="novel_relationships.csv",
                mime="text/csv"
            )

            # Clean up
            os.remove(temp_file_path)

# Add footer
st.markdown("---")
st.markdown("Novel Knowledge Graph Generator | Built with Streamlit, spaCy, and HuggingFace Transformers")