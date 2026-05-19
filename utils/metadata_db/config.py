# config.py
# Neo4j connection settings for the DICOM metadata ingestion pipeline.
# Update NEO4J_PASSWORD to whatever you set via cypher-shell.

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "neo4j-admin"   # ← change this
NEO4J_DATABASE = "neo4j"