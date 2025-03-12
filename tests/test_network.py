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
def cross_chat_data():
    """Fixture providing sample data for cross-chat participation analysis."""
    return pd.DataFrame({
        'sender': ['user1', 'user2', 'user2', 'user3', 'user3', 'user4'],
        'chat_id': ['chat1', 'chat1', 'chat2', 'chat2', 'chat3', 'chat3']
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

def test_analyze_cross_chat_participation(network_analysis, cross_chat_data):
    """
    Test the _analyze_cross_chat_participation method with sample data.
    Verifies bridge users (senders in multiple chats) and Jaccard similarity between chat pairs.
    """
    result = network_analysis._analyze_cross_chat_participation(cross_chat_data)
    
    # Expected outputs
    expected_bridge_users = {'user2': 2, 'user3': 2}
    expected_chat_similarity = {
        ("chat1", "chat2"): pytest.approx(1/3),
        ("chat1", "chat3"): 0,
        ("chat2", "chat3"): pytest.approx(1/3)
    }
    
    # Assertions
    assert result['bridge_users'] == expected_bridge_users, "Bridge users do not match expected output"
    assert result['chat_similarity'] == expected_chat_similarity, "Chat similarities do not match expected output"

def test_calculate_influence_metrics(network_analysis, sample_data):
    """
    Test the _calculate_influence_metrics method.
    Verifies total messages and chat participation counts for senders.
    """
    # Create bipartite graph from sample data
    G = network_analysis._create_bipartite_graph(sample_data)
    
    # Calculate influence metrics
    result = network_analysis._calculate_influence_metrics(G)
    
    # Expected results based on sample_data:
    # - user1: 3 messages (2 in chat1, 1 in chat2), 2 chats
    # - user2: 1 message (1 in chat1), 1 chat
    # - user3: 1 message (1 in chat2), 1 chat
    expected_influence = {
        "sender_influence": {
            "user1": {"total_messages": 3.0, "chats_participated": 2},
            "user2": {"total_messages": 1.0, "chats_participated": 1},
            "user3": {"total_messages": 1.0, "chats_participated": 1},
        }
    }
    
    # Assertions
    assert "sender_influence" in result, "Result should contain 'sender_influence' key"
    assert result == expected_influence, (
        f"Expected influence metrics {expected_influence}, "
        f"but got {result}"
    )
    
    # Verify sender nodes only (no chat nodes)
    sender_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    assert set(result["sender_influence"].keys()) == sender_nodes, (
        "Influence metrics should only include sender nodes"
    )
    
    # Verify type consistency
    for sender, metrics in result["sender_influence"].items():
        assert isinstance(metrics["total_messages"], float), "Total messages should be float"
        assert isinstance(metrics["chats_participated"], int), "Chats participated should be int"
        
def test_analyze(network_analysis, sample_data):
    """Test the full analyze method with weights and influence metrics integrated."""
    result = network_analysis.analyze(sample_data)
    # Basic structure checks
    assert isinstance(result, dict)
    assert "bipartite_graph" in result
    assert "sender_centrality" in result
    assert "chat_centrality" in result
    assert "communities" in result
    assert "community_metrics" in result
    assert "sender_projection" in result
    assert "sender_influence" in result
    
    # Validate specific outputs
    assert result["bipartite_graph"]["user1"]["chat1"]["weight"] == 2
    assert result["sender_centrality"]["degree"]["user1"] == 0.5
    
    # Validate influence metrics
    assert result["sender_influence"]["user1"]["total_messages"] == 3.0
    assert result["sender_influence"]["user1"]["chats_participated"] == 2
    assert result["sender_influence"]["user2"]["total_messages"] == 1.0
    assert result["sender_influence"]["user2"]["chats_participated"] == 1
    assert result["sender_influence"]["user3"]["total_messages"] == 1.0
    assert result["sender_influence"]["user3"]["chats_participated"] == 1

# New Tests for Edge Cases and Robustness
def test_analyze_empty_data(network_analysis):
    """Test analyze method with empty data."""
    empty_data = pd.DataFrame(columns=["sender", "chat_id", "content"])
    result = network_analysis.analyze(empty_data)
    assert isinstance(result, dict)
    assert result["bipartite_graph"].number_of_nodes() == 0
    assert result["bipartite_graph"].number_of_edges() == 0
    assert result["sender_centrality"] == {}
    assert result["chat_centrality"] == {}
    assert result["communities"] == {}
    assert result["community_metrics"]["densities"] == {}
    assert result["community_metrics"]["modularity"] == 0
    assert result["community_metrics"]["num_communities"] == 0
    assert result["community_metrics"]["sizes"] == {}
    assert result["sender_projection"].number_of_nodes() == 0
    assert result["sender_influence"] == {}

def test_analyze_single_chat(network_analysis):
    """Test analyze method with data from a single chat."""
    single_chat_data = pd.DataFrame({
        "sender": ["user1", "user2"],
        "chat_id": ["chat1", "chat1"],
        "content": ["Hi", "Hello"]
    })
    result = network_analysis.analyze(single_chat_data)
    assert result["bipartite_graph"].number_of_nodes() == 3  # 2 senders + 1 chat
    assert result["bipartite_graph"].number_of_edges() == 2
    assert result["sender_influence"]["user1"]["total_messages"] == 1.0
    assert result["sender_influence"]["user1"]["chats_participated"] == 1
    assert result["sender_influence"]["user2"]["total_messages"] == 1.0
    assert result["sender_influence"]["user2"]["chats_participated"] == 1