# config.py
# Neo4j connection settings for the DICOM metadata ingestion pipeline.
# Set NEO4J_PASSWORD as an environment variable or update here for local-only dev.

import os

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")  # set via environment variable
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")