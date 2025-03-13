import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import cast

import community as community_louvain
import networkx as nx
import pandas as pd
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
        self.logger.debug("Initialized NetworkAnalysis strategy with name: %s", name)

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
            - 'sender_influence': Influence metrics for senders.
            - 'cross_chat_metrics': Metrics on cross-chat participation.
            - 'reaction_graph': Directed graph of reactions.
            - 'reaction_metrics': Centrality metrics based on reactions.

        Notes
        -----
        Logs the start and completion of the analysis process.
        """
        self.logger.debug("Starting network analysis with data shape: %s", data.shape)

        # Create bipartite graph
        G = self._create_bipartite_graph(data)

        # Get node sets
        sender_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
        chat_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
        self.logger.debug(
            "Identified %d sender nodes and %d chat nodes", len(sender_nodes), len(chat_nodes)
        )

        # Compute centrality measures
        centrality_metrics = self._compute_centrality_measures(G, sender_nodes, chat_nodes)

        # Project sender nodes
        sender_projection = (
            nx.bipartite.weighted_projected_graph(G, sender_nodes) if sender_nodes else nx.Graph()
        )

        # Identify communities
        community_data = self._identify_communities(sender_projection)

        # Calculate influence metrics
        influence_metrics = self._calculate_influence_metrics(G, sender_nodes)

        # Analyze cross-chat participation
        cross_chat_metrics = self._analyze_cross_chat_participation(data)

        # Reaction analysis
        reaction_graph = self._create_reaction_graph(data)
        reaction_metrics = self._compute_reaction_metrics(reaction_graph)

        result: NetworkAnalysisResult = {
            "bipartite_graph": G,
            "sender_centrality": centrality_metrics["sender_centrality"],
            "chat_centrality": centrality_metrics["chat_centrality"],
            "communities": community_data["communities"],
            "community_metrics": community_data["community_metrics"],
            "sender_projection": sender_projection,
            "sender_influence": influence_metrics["sender_influence"],
            "cross_chat_metrics": cross_chat_metrics,
            "reaction_graph": reaction_graph,
            "reaction_metrics": reaction_metrics,
        }

        self.logger.debug(
            "Network analysis completed: %d senders, %d chats", len(sender_nodes), len(chat_nodes)
        )
        return result

    def _create_bipartite_graph(self, data: pd.DataFrame) -> nx.Graph:
        """
        Create a bipartite graph from Instagram message data.

        Parameters
        ----------
        data : pd.DataFrame
        DataFrame with 'sender', 'chat_id', and 'content' columns.

        Returns
        -------
        nx.Graph
        Bipartite graph with senders (bipartite=0) and chats (bipartite=1), weighted edges,
        and average message length attributes.

        Notes
        -----
        Edge weights represent the number of messages between sender and chat.
        Average message length is added as an edge attribute.
        """
        G: nx.Graph = nx.Graph()

        # Create a copy to avoid modifying the original DataFrame
        data_copy = data.copy()
        
        # Add sender nodes
        senders = data_copy["sender"].unique()
        G.add_nodes_from(senders, bipartite=0, type="sender")

        # Add chat nodes
        chats = data_copy["chat_id"].unique()
        G.add_nodes_from(chats, bipartite=1, type="chat")

        # Add weighted edges
        edge_weights = data_copy.groupby(["sender", "chat_id"]).size().reset_index(name="weight")
        G.add_weighted_edges_from(edge_weights[["sender", "chat_id", "weight"]].to_numpy())

        # Add average message length
        data_copy["msg_length"] = data_copy["content"].str.len()
        avg_lengths = data_copy.groupby(["sender", "chat_id"])["msg_length"].mean().reset_index()
        for row in avg_lengths.itertuples():
            if G.has_edge(row.sender, row.chat_id):
                nx.set_edge_attributes(G, {(row.sender, row.chat_id): {"avg_length": row.msg_length}})

        self.logger.debug(
            "Created bipartite graph: %d senders, %d chats, %d edges",
            len(senders),
            len(chats),
            G.number_of_edges(),
        )
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
        if G.number_of_nodes() == 0:
            self.logger.warning("Graph is empty, returning empty centrality metrics")
            return {"sender_centrality": {}, "chat_centrality": {}}

        self.logger.debug("Starting centrality measures computation")
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight="weight")

        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight="weight", max_iter=1000)
        except ArpackNoConvergence:
            self.logger.warning("Eigenvector centrality failed to converge, using unweighted")
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G, max_iter=1000)

        pagerank = nx.pagerank(G, weight="weight")
        closeness_centrality = nx.closeness_centrality(G)

        centrality_measures = {
            "degree": degree_centrality,
            "betweenness": betweenness_centrality,
            "eigenvector": eigenvector_centrality,
            "pagerank": pagerank,
            "closeness": closeness_centrality,
        }

        # Filter results for sender and chat nodes
        sender_centrality = {
            metric: {n: values[n] for n in sender_nodes} for metric, values in centrality_measures.items()
        }
        chat_centrality = {
            metric: {n: values[n] for n in chat_nodes} for metric, values in centrality_measures.items()
        }

        self.logger.debug(
            "Centrality measures computed: %d senders, %d chats",
            len(sender_centrality["degree"]),
            len(chat_centrality["degree"]),
        )
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
            self.logger.warning("Sender projection has no edges, returning empty communities.")
            return {
                "communities": {},
                "community_metrics": {"num_communities": 0, "sizes": {}, "modularity": 0, "densities": {}},
            }

        # Check for edge weights
        if not nx.get_edge_attributes(sender_projection, "weight"):
            self.logger.warning(
                "Sender projection graph edges do not have 'weight' attribute. Louvain algorithm might not use weights."
            )

        # Apply Louvain algorithm with weights
        partition = community_louvain.best_partition(sender_projection, weight="weight")

        communities = {node: community_id for node, community_id in partition.items()}

        # Compute community metrics
        sizes = Counter(partition.values())
        num_communities = len(sizes)
        densities = {
            community_id: nx.density(sender_projection.subgraph(community))  # type: ignore[no-untyped-call]
            for community_id, community in enumerate(partition)
        }
        modularity = community_louvain.modularity(partition, sender_projection, weight="weight")

        community_metrics = {
            "num_communities": num_communities,
            "sizes": sizes,
            "modularity": modularity,
            "densities": densities,
        }

        self.logger.debug("Detected %d communities with modularity: %.3f", num_communities, modularity)
        return {"communities": communities, "community_metrics": community_metrics}

    def _calculate_influence_metrics(
        self, G: nx.Graph, sender_nodes: set[str]
    ) -> dict[str, dict[str, dict[str, int]]]:
        """
        Calculate influence metrics for senders without temporal dependency.

        Parameters
        ----------
        G : nx.Graph
            Bipartite graph of senders and chats, where edges represent message activity
            and have a 'weight' attribute for message counts.
        sender_nodes : set[str]
            Set of sender node identifiers (strings) to analyze.

        Returns
        -------
        dict
            Dictionary with key 'sender_influence', mapping sender nodes to:
            - 'total_messages': int
                Total number of messages sent by the sender across all chats (weighted degree).
            - 'chats_participated': int
                Number of chats the sender participates in (unweighted degree).
        """
        sender_influence = {
            sender: {
                "total_messages": cast(int, G.degree(sender, weight="weight")),
                "chats_participated": cast(int, G.degree(sender)),
            }
            for sender in sender_nodes
        }

        self.logger.debug("Influence metrics calculated for %d senders", len(sender_influence))
        return {"sender_influence": sender_influence}

    def _analyze_cross_chat_participation(self, data: pd.DataFrame) -> dict:
        """
        Analyze cross-chat participation patterns.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'sender' and 'chat_id' columns.

        Returns
        -------
        dict
            Dictionary with two keys:
            - 'bridge_users': Mapping of senders to number of chats (for those in >1 chat).
            - 'chat_similarity': Mapping of chat pairs to Jaccard similarity of their senders.
        """
        self.logger.debug("Analyzing cross-chat participation")
        participation = data.groupby(["sender", "chat_id"]).size().unstack(fill_value=0)

        chat_similarity = {}
        chats = data["chat_id"].unique()
        for chat1, chat2 in combinations(chats, 2):
            chat_users1 = set(data[data["chat_id"] == chat1]["sender"].unique())
            chat_users2 = set(data[data["chat_id"] == chat2]["sender"].unique())
            intersection = len(chat_users1.intersection(chat_users2))
            union = len(chat_users1.union(chat_users2))
            similarity = intersection / union if union > 0 else 0
            chat_similarity[(chat1, chat2)] = similarity

        user_chat_counts = data.groupby("sender")["chat_id"].nunique().sort_values(ascending=False)
        bridge_users = user_chat_counts[user_chat_counts > 1].to_dict()

        self.logger.debug(
            "Found %d bridge users and %d chat similarity pairs", len(bridge_users), len(chat_similarity)
        )
        return {"bridge_users": bridge_users, "chat_similarity": chat_similarity}

    def _create_reaction_graph(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Create a directed graph where edges represent reactions from reactors to senders.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'sender' column and optionally 'reactions' column.
            'reactions' contains lists of (reaction_type, reactor) tuples.

        Returns
        -------
        nx.DiGraph
            Directed graph with edges from reactor to sender, weighted by number of reactions.
            If 'reactions' column is missing or no valid reactions exist, returns an empty graph
            with sender nodes (if available).
        """
        # Initialize directed graph
        reaction_graph = nx.DiGraph()

        # Add sender nodes if 'sender' column exists
        if "sender" in data.columns:
            senders = set(data["sender"])
            reaction_graph.add_nodes_from(senders)
        else:
            self.logger.warning("No 'sender' column, returning empty reaction graph")
            return reaction_graph  # Return empty graph if no senders

        # Check if 'reactions' column exists
        if "reactions" not in data.columns:
            self.logger.warning(
                "No 'reactions' column, returning graph with %d sender nodes",
                reaction_graph.number_of_nodes(),
            )
            return reaction_graph  # Return graph with only sender nodes if no reactions column

        # Filter data to only include rows with non-empty reactions
        data_with_reactions = data[data["reactions"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

        if data_with_reactions.empty:
            self.logger.warning(
                "No valid reactions, returning graph with %d sender nodes", reaction_graph.number_of_nodes()
            )
            return reaction_graph  # Return graph with no sender nodes if no valid reactions

        # Explode reactions and extract reactors
        exploded = data_with_reactions[["sender", "reactions"]].explode("reactions")
        exploded["reactor"] = exploded["reactions"].apply(
            lambda x: x[1] if isinstance(x, list | tuple) and len(x) > 1 else None
        )
        exploded = exploded.dropna(subset=["reactor"])

        # Get all unique reactors
        reactors: set[str] = set(exploded["reactor"])
        all_users = senders | reactors

        # Initialie directed graph
        reaction_graph = nx.DiGraph()
        reaction_graph.add_nodes_from(all_users)

        # Count reactions per (reactor, sender) pair
        reaction_counts = exploded.groupby(["reactor", "sender"]).size().to_dict()

        # Add weighted edges
        for (reactor, sender), count in reaction_counts.items():
            reaction_graph.add_edge(reactor, sender, weight=count)

        self.logger.debug(
            "Created reaction graph: %d nodes, %d edges",
            reaction_graph.number_of_nodes(),
            reaction_graph.number_of_edges(),
        )
        return reaction_graph

    def _compute_reaction_metrics(self, reaction_graph: nx.DiGraph) -> dict:
        """
        Compute centrality metrics on the reaction graph.

        Parameters
        ----------
        reaction_graph : nx.DiGraph
            Directed graph of reactions.

        Returns
        -------
        dict
            Dictionary with:
            - 'in_degree': In-degree centrality (reactions received).
            - 'out_degree': Out-degree centrality (reactions given).
            - 'pagerank': PageRank scores based on reactions.
            Returns empty dicts if the graph has no edges (i.e., no reactions).
        """
        if reaction_graph.number_of_edges() == 0:
            self.logger.warning("Reaction graph has no edges, returning empty metrics")
            return {"in_degree": {}, "out_degree": {}, "pagerank": {}}

        self.logger.debug("Computing reaction metrics")
        in_degree = nx.in_degree_centrality(reaction_graph)
        out_degree = nx.out_degree_centrality(reaction_graph)
        pagerank = nx.pagerank(reaction_graph, weight="weight")
        self.logger.debug("Reaction metrics computed: in_degree, out_degree, pagerank")

        return {"in_degree": in_degree, "out_degree": out_degree, "pagerank": pagerank}

    def save_results(self, results: NetworkAnalysisResult, output_dir: Path) -> None:
        """
        Save network analysis results to disk.

        Parameters
        ----------
        results : NetworkAnalysisResult
            Dictionary containing analysis results from `analyze`.
        output_dir : Path
            Directory path where results will be saved.

        Notes
        -----
        Creates a subdirectory named after the strategy and saves results as CSV and JSON files.
        Logs the save location upon completion.
        """
        strategy_dir = output_dir / self.name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Sender centrality
        sender_centrality_df = pd.DataFrame(results["sender_centrality"]).T
        sender_centrality_df.index.name = "sender"
        sender_centrality_df.to_csv(strategy_dir / "sender_centrality.csv")
        self.logger.debug("Saved sender_centrality.csv")

        # Chat centrality
        chat_centrality_df = pd.DataFrame(results["chat_centrality"]).T
        chat_centrality_df.index.name = "chat_id"
        chat_centrality_df.to_csv(strategy_dir / "chat_centrality.csv")
        self.logger.debug("Saved chat_centrality.csv")

        # Community metrics
        with (strategy_dir / "community_metrics.json").open("w") as file:
            json.dump(results["community_metrics"], file)
        self.logger.debug("Saved community_metrics.json")

        # Influence metrics
        pd.DataFrame(results["sender_influence"]).T.to_csv(strategy_dir / "sender_influence.csv")
        self.logger.debug("Saved sender_influence.csv")

        # Cross-chat metrics
        pd.DataFrame.from_dict(
            results["cross_chat_metrics"]["bridge_users"], orient="index", columns=["chat_count"]
        ).to_csv(strategy_dir / "bridge_users.csv")
        self.logger.debug("Saved bridge_users.csv")
        pd.DataFrame.from_dict(
            results["cross_chat_metrics"]["chat_similarity"], orient="index", columns=["similarity"]
        ).to_csv(strategy_dir / "chat_similarity.csv")
        self.logger.debug("Saved chat_similarity.csv")

        # Save reaction metrics
        pd.DataFrame(results["reaction_metrics"]).T.to_csv(strategy_dir / "reaction_centrality.csv")
        self.logger.debug("Saved reaction_centrality.csv")

        self.logger.info("Saved network analysis results to %s", strategy_dir)
