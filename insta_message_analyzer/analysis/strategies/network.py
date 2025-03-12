from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import community as community_louvain
from scipy.sparse.linalg import ArpackNoConvergence

from insta_message_analyzer.analysis.protocol import AnalysisStrategy
from insta_message_analyzer.analysis.types import NetworkAnalysisResult
from insta_message_analyzer.utils.logging import get_logger


class NetworkAnalysis(AnalysisStrategy):
    """Analyzes sender-chat interactions as a bipartite netwrok, focusing on structural and relational metrics."""

    def __init__(
        self,
        name: str = "NetworkAnalysis",
    ) -> None:
        """
        Initialize the NetworkAnalysis strategy with a logger.

        Notes
        -----
        The logger is configured using the module's name for tracking analysis steps.
        """
        self.logger = get_logger(__name__)
        self._name = name

    @property
    def name(self) -> str:
        """Get the unique name of the strategy.

        Returns
        -------
        str
            The name of the strategy instance.
        """
        return self._name

    def analyze(self, data: pd.DataFrame) -> NetworkAnalysisResult:
        """
        Perform network analysis on Instagram message data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing 'sender', 'chat_id', 'reactions', and optionally 'content' columns.

        Returns
        -------
        NetworkAnalysisResult
            Dictionary containing graph objects and metrics:
            - 'bipartite_graph': The bipartite NetworkX graph.
            - 'sender_centrality': Centrality metrics for senders.
            - 'chat_centrality': Centrality metrics for chats.
            - 'communities': Mapping of sender nodes to community IDs.
            - 'community_metrics': Metrics about detected communities.
            - 'sender_projection': Projected graph of senders.
            - 'influence_metrics': Influence metrics for senders.
            - 'cross_chat_metrics': Metrics on cross-chat participation.
            - 'reaction_metrics': Centrality metrics based on reactions.

        Notes
        -----
        Logs the start and completion of the analysis process.
        """
        self.logger.debug("Starting network analysis")

        # Create bipartite graph
        G = self._create_bipartite_graph(data)

        # Get node sets
        sender_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        chat_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}

        # Compute centrality measures
        self.logger.debug("Computing centrality measures")
        centrality_metrics = self._compute_centrality_measures(G, sender_nodes, chat_nodes)

        # Compute sender projection and communities (placeholders for full implementation)
        sender_projection = nx.bipartite.weighted_projected_graph(G, sender_nodes)
        community_data = self._identify_communities(sender_projection)

        # Placeholder for additional metrics
        influence_metrics = {}
        cross_chat_metrics = {}
        reaction_metrics = {}

        result: NetworkAnalysisResult = {
            "bipartite_graph": G,
            "sender_centrality": centrality_metrics["sender_centrality"],
            "chat_centrality": centrality_metrics["chat_centrality"],
            "communities": community_data["communities"],
            "community_metrics": community_data["community_metrics"],
            "sender_projection": sender_projection,
            "influence_metrics": influence_metrics,
            "cross_chat_metrics": cross_chat_metrics,
            "reaction_metrics": reaction_metrics,
        }

        self.logger.debug("Network analysis completed")
        return result

    def _create_bipartite_graph(self, data: pd.DataFrame) -> nx.Graph:
        """
        Create a bipartite graph from Instagram message data.

        Parameters
        ----------
        data : pd.DataFrame
        DataFrame with 'sender', 'chat_id', and optionally 'content' columns.

        Returns
        -------
        nx.Graph
        Bipartite graph with senders (bipartite=0) and chats (bipartite=1), weighted edges,
        and optional average message length attributes.

        Notes
        -----
        Edge weights represent the number of messages between sender and chat.
        If 'content' is present, average message length is added as an edge attribute.
        """
        G: nx.Graph = nx.Graph()

        # Add sender nodes
        senders = data["sender"].unique()
        G.add_nodes_from(senders, bipartite=0, type="sender")

        # Add chat nodes
        chats = data["chat_id"].unique()
        G.add_nodes_from(chats, bipartite=1, type="chat")

        # Add weighted edges
        edge_weights = data.groupby(["sender", "chat_id"]).size().reset_index(name="weight")
        for row in edge_weights.itertuples():
            G.add_edge(
                row.sender, row.chat_id, weight=row.weight
            )  # NOTE: this can be vectorized #type: ignore[reportArgumentType]

        # Add average message length
        data["msg_length"] = data["content"].str.len()
        avg_lengths = data.groupby(["sender", "chat_id"])["msg_length"].mean().reset_index()
        for row in avg_lengths.itertuples():
            if G.has_edge(row.sender, row.chat_id):
                nx.set_edge_attributes(G, {(row.sender, row.chat_id): {"avg_length": row.msg_length}})

        return G

    def _compute_centrality_measures(self, G: nx.Graph, sender_nodes: set, chat_nodes: set) -> dict:
        """
        Compute centrality measures for nodes in the bipartite graph.

        Parameters
        ----------
        G : nx.Graph
            Bipartite graph of senders and chats.
        sender_nodes : set
            Set of sender node identifiers.
        chat_nodes : set
            Set of chat node identifiers.

        Returns
        -------
        dict
            Dictionary with two keys:
            - 'sender_centrality': Centrality metrics for sender nodes.
            - 'chat_centrality': Centrality metrics for chat nodes.
            Each contains sub-dictionaries for degree, betweenness, eigenvector, pagerank,
            and closeness centrality.

        Notes
        -----
        Uses weighted measures where applicable; falls back to unweighted eigenvector centrality
        if convergence fails.
        """
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight="weight")

        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight="weight", max_iter=1000)
        except ArpackNoConvergence:
            self.logger.warning("Eigenvector centrality failed to converge, using unweighted")
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G, max_iter=1000)

        pagerank = nx.pagerank(G, weight="weight")
        closeness_centrality = nx.closeness_centrality(G)

        centrality_meaures = {
            "degree": degree_centrality,
            "betweenness": betweenness_centrality,
            "eigenvector": eigenvector_centrality,
            "pagerank": pagerank,
            "closeness": closeness_centrality,
        }

        # Filter results for sender and chat nodes
        sender_centrality = {
            metric: {n: values[n] for n in sender_nodes} for metric, values in centrality_meaures.items()
        }
        chat_centrality = {
            metric: {n: values[n] for n in chat_nodes} for metric, values in centrality_meaures.items()
        }

        return {"sender_centrality": sender_centrality, "chat_centrality": chat_centrality}

    def _identify_communities(self, sender_projection: nx.Graph) -> dict:
        """
        Identify communities in the sender projection using the Louvain algorithm.

        Parameters
        ----------
        sender_projection : nx.Graph
        Weighted projected graph of senders.

        Returns
        -------
        dict
        Dictionary with two keys:
        - 'communities': Mapping of sender nodes to community IDs (int).
        - 'community_metrics': Dictionary with:
            - 'num_communities': Number of communities.
            - 'sizes': Mapping of community IDs to their sizes.
            - 'modularity': Modularity score of the partition.
            - 'densities': Mapping of community IDs to subgraph densities.

        Notes
        -----
        Uses the Louvain algorithm from the community package, considering edge weights.
        """
        # Verify that the 'sender_projection' graph has edges
        if not sender_projection.number_of_edges():
            self.logger.warning("Sender projecting has no edges, skipping community detection.")
            return {"communities": {}, "community_metrics": {}}

        # Check for edge weights
        if not nx.get_edge_attributes(sender_projection, "weight"):
            self.logger.warning(
                "Sender projection graph edges do not have 'weight' attribute. Leiden algorithm might not use weights."
            )

        # Apply Louvain algorithm with weights
        partition = community_louvain.best_partition(sender_projection, weight="weight")

        # Build communities list from partition
        communites_dict = defaultdict(list)
        for node, comm_id in partition.items():
            communites_dict[comm_id].append(node)
        communities_list = list(communites_dict.values())

        # Assign consecutive community IDs starting from 0
        communities = {
            node: community_id
            for community_id, community in enumerate(communities_list)
            for node in community
        }

        # Compute community metrics
        num_communities = len(communities_list)
        sizes = {community_id: len(community) for community_id, community in enumerate(communities_list)}
        densities = {
            community_id: nx.density(sender_projection.subgraph(community)) #type: ignore[no-untyped-call]
            for community_id, community in enumerate(communities_list)
        }
        modularity = community_louvain.modularity(partition, sender_projection, weight="weight")

        community_metrics = {
            "num_communities": num_communities,
            "sizes": sizes,
            "modularity": modularity,
            "densities": densities,
        }

        return {"communities": communities, "community_metrics": community_metrics}

    def save_results(self, results: NetworkAnalysisResult, output_dir: Path) -> None:
        return None
