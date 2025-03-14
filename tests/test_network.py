import pytest
import pandas as pd
import networkx as nx
from unittest.mock import Mock

from insta_message_analyzer.analysis.strategies.network import NetworkAnalysis
from insta_message_analyzer.analysis.analysis_types import NetworkAnalysisResult

# ---------------------------
# Fixtures for sample data
# ---------------------------


@pytest.fixture
def sample_data():
    """Fixture providing sample Instagram message data."""
    return pd.DataFrame(
        {
            "sender": ["user1", "user2", "user1", "user3", "user1"],
            "chat_id": ["chat1", "chat1", "chat1", "chat2", "chat2"],
            "content": ["Hi", "Hello", "Hey there", "Yo", "Another one"],
        }
    )


@pytest.fixture
def cross_chat_data():
    """Fixture providing sample data for cross-chat participation analysis."""
    return pd.DataFrame(
        {
            "sender": ["user1", "user2", "user2", "user3", "user3", "user4"],
            "chat_id": ["chat1", "chat1", "chat2", "chat2", "chat3", "chat3"],
        }
    )


@pytest.fixture
def reaction_data():
    """Fixture providing sample Instagram reaction data."""
    return pd.DataFrame(
        {
            "sender": ["user1", "user2", "user1", "user3"],
            "reactions": [
                [("like", "user2"), ("love", "user3")],  # user2 and user3 react to user1
                [("like", "user1")],  # user1 reacts to user2
                [("haha", "user2")],  # user2 reacts to user1 again
                [],  # no reactions to user3
            ],
        }
    )


@pytest.fixture
def network_analysis():
    """Fixture providing a NetworkAnalysis instance with a mock logger."""
    logger = Mock()
    return NetworkAnalysis()


# ---------------------------
# Tests for bipartite graph creation
# ---------------------------


@pytest.mark.bipartite
def test_create_bipartite_graph(network_analysis, sample_data):
    # Act: create the bipartite graph
    graph = network_analysis._create_bipartite_graph(sample_data)

    # Assert: check type and expected node/edge counts
    assert isinstance(graph, nx.Graph), "Graph must be a networkx Graph instance"
    # Expecting 3 sender nodes and 2 chat nodes
    assert graph.number_of_nodes() == 5, "Graph should have 5 nodes (3 senders + 2 chats)"
    assert graph.number_of_edges() == 4, "Graph should have 4 unique sender-chat edges"

    # Assert: check weights on specific edges using clear messages
    assert graph["user1"]["chat1"]["weight"] == 2, (
        f"Expected weight 2 for edge 'user1'-'chat1', got {graph['user1']['chat1']['weight']}"
    )
    assert graph["user1"]["chat2"]["weight"] == 1, (
        f"Expected weight 1 for edge 'user1'-'chat2', got {graph['user1']['chat2']['weight']}"
    )
    assert graph["user2"]["chat1"]["weight"] == 1, (
        f"Expected weight 1 for edge 'user2'-'chat1', got {graph['user2']['chat1']['weight']}"
    )
    assert graph["user3"]["chat2"]["weight"] == 1, (
        f"Expected weight 1 for edge 'user3'-'chat2', got {graph['user3']['chat2']['weight']}"
    )


# ---------------------------
# Tests for centrality measures
# ---------------------------


@pytest.mark.centrality
def test_compute_centrality_measures(network_analysis, sample_data):
    """Test the _compute_centrality_measures method."""
    # Arrange: create graph and extract node sets
    graph = network_analysis._create_bipartite_graph(sample_data)
    sender_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
    chat_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 1}

    # Act: compute centrality measures
    result = network_analysis._compute_centrality_measures(graph, sender_nodes, chat_nodes)

    # Assert: check expected keys and normalized values using pytest.approx
    assert "sender_centrality" in result, "Result should include 'sender_centrality'"
    sender_centrality = result["sender_centrality"]
    assert "degree" in sender_centrality, "Degree centrality missing in sender centrality"
    assert "user1" in sender_centrality["degree"], "user1 missing in degree centrality"
    assert "betweenness" in sender_centrality, "Weighted betweenness centrality missing"
    assert "eigenvector" in sender_centrality, "Weighted eigenvector centrality missing"
    # Using approx to compare float values
    assert sender_centrality["degree"]["user1"] == pytest.approx(0.5), (
        "Normalized degree centrality of user1 should be approximately 0.5"
    )


# ---------------------------
# Tests for community detection
# ---------------------------


@pytest.mark.communities
def test_identify_communities(network_analysis, sample_data):
    """Test the _identify_communities method with weighted projection."""
    # Arrange: create bipartite graph and project sender nodes
    graph = network_analysis._create_bipartite_graph(sample_data)
    sender_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
    sender_projection = nx.bipartite.weighted_projected_graph(graph, sender_nodes)

    # Act: detect communities
    result = network_analysis._identify_communities(sender_projection)

    # Assert: check for community structure and modularity computation
    assert "communities" in result, "Result should include 'communities'"
    assert "community_metrics" in result, "Result should include 'community_metrics'"
    metrics = result["community_metrics"]
    assert metrics["num_communities"] > 0, "There should be at least one community detected"
    assert "modularity" in metrics, "Modularity should be computed with weights"


# ---------------------------
# Tests for cross-chat participation analysis
# ---------------------------


@pytest.mark.cross_chat
def test_analyze_cross_chat_participation(network_analysis, cross_chat_data):
    """
    Test the _analyze_cross_chat_participation method with sample data.
    Verifies bridge users (senders in multiple chats) and Jaccard similarity between chat pairs.
    """
    # Act: analyze cross-chat participation
    result = network_analysis._analyze_cross_chat_participation(cross_chat_data)

    # Expected outputs
    expected_bridge_users = {"user2": 2, "user3": 2}
    expected_chat_similarity = {
        ("chat1", "chat2"): pytest.approx(1 / 3),
        ("chat1", "chat3"): 0,
        ("chat2", "chat3"): pytest.approx(1 / 3),
    }

    # Assert: check both bridge users and similarity metrics
    assert result["bridge_users"] == expected_bridge_users, "Bridge users do not match expected output"
    assert result["chat_similarity"] == expected_chat_similarity, (
        "Chat similarities do not match expected output"
    )


# ---------------------------
# Tests for influence metrics
# ---------------------------


@pytest.mark.influence
def test_calculate_influence_metrics(network_analysis, sample_data):
    """
    Test the _calculate_influence_metrics method.
    Verifies total messages and chat participation counts for senders.
    """
    # Arrange: create graph from sample data
    graph = network_analysis._create_bipartite_graph(sample_data)
    sender_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}

    # Act: calculate influence metrics
    result = network_analysis._calculate_influence_metrics(graph, sender_nodes)

    # Expected results based on sample_data:
    expected_influence = {
        "sender_influence": {
            "user1": {"total_messages": 3.0, "chats_participated": 2},
            "user2": {"total_messages": 1.0, "chats_participated": 1},
            "user3": {"total_messages": 1.0, "chats_participated": 1},
        }
    }

    # Assert: basic structure and value comparisons
    assert "sender_influence" in result, "Result should contain 'sender_influence'"
    assert result == expected_influence, (
        f"Expected influence metrics {expected_influence}, but got {result}"
    )

    # Additional assertions for type consistency
    for sender, metrics in result["sender_influence"].items():
        assert isinstance(metrics["total_messages"], int), "Total messages should be int"
        assert isinstance(metrics["chats_participated"], int), "Chats participated should be int"


# ---------------------------
# Tests for reaction graph and metrics
# ---------------------------


@pytest.mark.reaction_graph
def test_create_reaction_graph(network_analysis, reaction_data):
    """Test the _create_reaction_graph method."""
    # Act: create reaction graph
    reaction_graph = network_analysis._create_reaction_graph(reaction_data)

    # Assert: check type and expected nodes/edges
    assert isinstance(reaction_graph, nx.DiGraph), "Reaction graph should be a directed graph"
    expected_nodes = {"user1", "user2", "user3"}
    assert set(reaction_graph.nodes) == expected_nodes, (
        f"Expected nodes {expected_nodes}, got {set(reaction_graph.nodes)}"
    )
    # Check edges and weights (reactor -> sender)
    assert reaction_graph.has_edge("user2", "user1"), "Edge user2 -> user1 missing"
    assert reaction_graph["user2"]["user1"]["weight"] == 2, "Edge weight user2 -> user1 should be 2"
    assert reaction_graph.has_edge("user3", "user1"), "Edge user3 -> user1 missing"
    assert reaction_graph["user3"]["user1"]["weight"] == 1, "Edge weight user3 -> user1 should be 1"
    assert reaction_graph.has_edge("user1", "user2"), "Edge user1 -> user2 missing"
    assert reaction_graph["user1"]["user2"]["weight"] == 1, "Edge weight user1 -> user2 should be 1"
    assert len(reaction_graph.edges) == 3, "Reaction graph should have exactly 3 edges"


@pytest.mark.reaction_metrics
def test_compute_reaction_metrics(network_analysis, reaction_data):
    """Test the _compute_reaction_metrics method."""
    # Arrange: create reaction graph
    reaction_graph = network_analysis._create_reaction_graph(reaction_data)

    # Act: compute reaction metrics
    metrics = network_analysis._compute_reaction_metrics(reaction_graph)

    # Assert: check existence of all expected metric keys
    for key in ("in_degree", "out_degree", "pagerank"):
        assert key in metrics, f"{key} centrality missing in reaction metrics"

    # Verify in-degree centrality with normalized calculations (n-1 where n=3)
    assert metrics["in_degree"]["user1"] == pytest.approx(1.0), "Incorrect in-degree for user1"
    assert metrics["in_degree"]["user2"] == pytest.approx(0.5), "Incorrect in-degree for user2"
    assert metrics["in_degree"]["user3"] == pytest.approx(0.0), "Incorrect in-degree for user3"

    # Verify out-degree centrality
    assert metrics["out_degree"]["user1"] == pytest.approx(0.5), "Incorrect out-degree for user1"
    assert metrics["out_degree"]["user2"] == pytest.approx(0.5), "Incorrect out-degree for user2"
    assert metrics["out_degree"]["user3"] == pytest.approx(0.5), "Incorrect out-degree for user3"

    # Compare PageRank with networkx calculation using pytest.approx
    expected_pagerank = nx.pagerank(reaction_graph, weight="weight")
    for user in ["user1", "user2", "user3"]:
        assert metrics["pagerank"][user] == pytest.approx(expected_pagerank[user]), (
            f"Incorrect PageRank for {user}"
        )


@pytest.mark.integration
def test_analyze(network_analysis, sample_data):
    """Test the full analyze method with weights and influence metrics integrated."""
    # Act: perform full analysis using public 'analyze' method
    result = network_analysis.analyze(sample_data)

    # Ensure result follows the expected structure
    assert isinstance(result, dict), "Analyze should return a dictionary"

    # Validate specific keys exist
    assert set(result.keys()) == set(NetworkAnalysisResult.__annotations__.keys()), (
        "Analyze output keys mismatch"
    )

    # Validate selected outputs with clear assertions
    assert result["bipartite_graph"]["user1"]["chat1"]["weight"] == 2, (
        "Bipartite graph edge weight for user1-chat1 should be 2"
    )
    assert result["sender_centrality"]["degree"]["user1"] == pytest.approx(0.5), (
        "Degree centrality for user1 should be approximately 0.5"
    )
    # Influence metrics
    influence = result["sender_influence"]
    assert influence["user1"]["total_messages"] == 3, "user1 should have 3 messages"
    assert influence["user1"]["chats_participated"] == 2, "user1 should participate in 2 chats"
    assert influence["user2"]["total_messages"] == 1, "user2 should have 1 message"
    assert influence["user2"]["chats_participated"] == 1, "user2 should participate in 1 chat"
    assert influence["user3"]["total_messages"] == 1, "user3 should have 1 message"
    assert influence["user3"]["chats_participated"] == 1, "user3 should participate in 1 chat"


# ---------------------------
# Tests for edge cases
# ---------------------------


@pytest.mark.edge_cases
@pytest.mark.parametrize(
    "data, expected_nodes, expected_edges",
    [
        (pd.DataFrame(columns=["sender", "chat_id", "content"]), 0, 0),
        (
            pd.DataFrame(
                {"sender": ["user1", "user2"], "chat_id": ["chat1", "chat1"], "content": ["Hi", "Hello"]}
            ),
            3,
            2,
        ),
    ],
)
def test_analyze_edge_cases(network_analysis, data, expected_nodes, expected_edges):
    # Act: analyze edge-case data
    result = network_analysis.analyze(data)

    # Assert: check bipartite graph node and edge counts
    bipartite_graph = result["bipartite_graph"]
    assert bipartite_graph.number_of_nodes() == expected_nodes, (
        f"Expected {expected_nodes} nodes, got {bipartite_graph.number_of_nodes()}"
    )
    assert bipartite_graph.number_of_edges() == expected_edges, (
        f"Expected {expected_edges} edges, got {bipartite_graph.number_of_edges()}"
    )

    # Additional checks for empty data case
    if expected_nodes == 0:
        assert result["sender_centrality"] == {}
        assert result["chat_centrality"] == {}
        assert result["communities"] == {}
        assert result["community_metrics"]["modularity"] == 0
        assert result["community_metrics"]["num_communities"] == 0
        assert result["community_metrics"]["densities"] == {}
        assert result["community_metrics"]["sizes"] == {}
        assert result["sender_projection"].number_of_nodes() == 0
        assert result["sender_influence"] == {}
