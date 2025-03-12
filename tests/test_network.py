import pytest
import pandas as pd
import networkx as nx
from unittest.mock import Mock

# Import NetworkAnalysis from your package
from insta_message_analyzer.analysis.strategies.network import NetworkAnalysis


@pytest.fixture
def sample_data():
    """Fixture providing sample Instagram message data."""
    return pd.DataFrame({
        "sender": ["user1", "user2", "user1", "user3", "user1"],
        "chat_id": ["chat1", "chat1", "chat1", "chat2", "chat2"],
        "content": ["Hi", "Hello", "Hey there", "Yo", "Another one"]
    })

@pytest.fixture
def network_analysis():
    """Fixture providing a NetworkAnalysis instance with a mock logger."""
    logger = Mock()  # Mock logger to avoid real logging during tests
    return NetworkAnalysis()

def test_create_bipartite_graph(network_analysis, sample_data):
    G = network_analysis._create_bipartite_graph(sample_data)
    print("Graph edges:", list(G.edges(data=True)))
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 5  # 3 senders + 2 chats
    assert G.number_of_edges() == 4  # 4 unique sender-chat pairs
    assert G["user1"]["chat1"]["weight"] == 2, f"Expected weight 2 for user1-chat1, got {G['user1']['chat1']['weight']}"
    assert G["user1"]["chat2"]["weight"] == 1, f"Expected weight 1 for user1-chat2, got {G['user1']['chat2']['weight']}"
    assert G["user2"]["chat1"]["weight"] == 1, f"Expected weight 1 for user2-chat1, got {G['user2']['chat1']['weight']}"
    assert G["user3"]["chat2"]["weight"] == 1, f"Expected weight 1 for user3-chat2, got {G['user3']['chat2']['weight']}"
    
def test_compute_centrality_measures(network_analysis, sample_data):
    """Test the _compute_centrality_measures method."""
    G = network_analysis._create_bipartite_graph(sample_data)
    sender_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    chat_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
    result = network_analysis._compute_centrality_measures(G, sender_nodes, chat_nodes)
    assert "sender_centrality" in result
    assert "degree" in result["sender_centrality"]
    assert "user1" in result["sender_centrality"]["degree"]
    # Check weighted centrality measures
    assert "betweenness" in result["sender_centrality"], "Weighted betweenness centrality missing"
    assert "eigenvector" in result["sender_centrality"], "Weighted eigenvector centrality missing"
    assert result["sender_centrality"]["degree"]["user1"] == 0.5, "Expected normalized degree centrality of 0.5 for user1"  
    
def test_identify_communities(network_analysis, sample_data):
    """Test the _identify_communities method with weighted projection."""
    G = network_analysis._create_bipartite_graph(sample_data)
    sender_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    sender_projection = nx.bipartite.weighted_projected_graph(G, sender_nodes)
    result = network_analysis._identify_communities(sender_projection)
    assert "communities" in result
    assert "community_metrics" in result
    assert result["community_metrics"]["num_communities"] > 0
    # Verify weights are considered
    assert "modularity" in result["community_metrics"], "Modularity should be computed with weights"


def test_analyze(network_analysis, sample_data):
    """Test the full analyze method with weights integrated."""
    result = network_analysis.analyze(sample_data)
    assert isinstance(result, dict)
    assert "bipartite_graph" in result
    assert "sender_centrality" in result
    assert "chat_centrality" in result
    assert "communities" in result
    assert "community_metrics" in result
    assert "sender_projection" in result
    # Validate weight-related outputs
    assert result["bipartite_graph"]["user1"]["chat1"]["weight"] == 2, "Bipartite graph weight incorrect"
    assert "degree" in result["sender_centrality"]
    assert result["sender_centrality"]["degree"]["user1"] == 0.5, "Sender centrality degree should reflect weighted edges"