from typing import List, Set, Dict, Tuple, Optional, Callable
from collections import Counter
from itertools import chain

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from orangecontrib.text.vectorization.sbert import SBERT
from Orange.util import dummy_callback, wrap_callback


class Tree:

    def __init__(self, adj_list: List[Set[int]], labels: List[str], root: int) -> None:
        assert (
            set([neighbor for node in adj_list for neighbor in node]) ==
            set(range(len(adj_list)))
        ), "Vertices must be enumerated from 0 to n."
        assert len(adj_list) == len(labels), "adj_list and labels must be of same length"
        self.adj_list = adj_list
        self.labels = labels
        self.root = root

    @classmethod
    def from_dict(cls, tree: Dict):
        assert len(tree.keys()) == 1, "Not a tree"

        def _from_dict(tree, root, edgelist, labels):
            labels.append(root)
            idx = len(labels) - 1
            for i, neighbor in enumerate(tree[root]):
                edgelist.append((idx, len(labels)))
                _from_dict(tree[root], neighbor, edgelist, labels)

        edgelist = list()
        labels = list()
        _from_dict(tree, list(tree.keys())[0], edgelist, labels)
        adj_list = [set() for _ in range(len(labels))]
        for u, v in edgelist:
            adj_list[u].add(v)
            adj_list[v].add(u)

        return Tree(adj_list, labels, 0)

    @classmethod
    def from_prufer_sequence(cls, sequence: List[int], labels: List[str], root: int = None):
        n_nodes = len(sequence) + 2
        adj_list = [set() for _ in range(n_nodes)]
        degrees = Counter(chain(sequence, range(n_nodes)))
        idx = u = next(i for i in range(n_nodes) if degrees[i] == 1)
        nodes = set(range(n_nodes))
        for v in sequence:
            nodes.remove(u)
            adj_list[u].add(v)
            adj_list[v].add(u)
            degrees[v] -= 1
            if v < idx and degrees[v] == 1:
                u = v
            else:
                idx = u = next(i for i in range(idx + 1, n_nodes) if degrees[i] == 1)
        u, v = nodes
        adj_list[u].add(v)
        adj_list[v].add(u)

        return Tree(adj_list, labels, root)

    def to_dict(self) -> Dict:

        def _to_dict(adj_list, labels, root, prev=None):
            tree = dict()
            neighbors = [node for node in adj_list[root] if node != prev]
            for neighbor in neighbors:
                tree[labels[neighbor]] = _to_dict(adj_list, labels, neighbor, prev=root)
            return tree

        return {self.labels[self.root]: _to_dict(self.adj_list, self.labels, self.root)}


class FitnessFunction:

    def __init__(
        self,
        words,
        similarities,
        min_children: int = 2,
        max_children: int = 7,
        parent_sim_weight: float = 0.5,
        sibling_sim_weight: float = 0.3,
        n_children_weight: float = 0.1,
        length_intersection_weight: float = 0.1,
    ):
        self.words = words
        self.sims = similarities
        self.min_children = min_children
        self.max_children = max_children
        self.parent_sim_weight = parent_sim_weight
        self.sibling_sim_weight = sibling_sim_weight
        self.n_children_weight = n_children_weight
        self.length_intersection_weight = length_intersection_weight

    def fitness(self, tree: Tree, root: Optional[int] = None) -> Tuple[float, int]:
        score_list = list()
        if root is None:
            root = np.argmin([
                np.min(self.sims[i, list(tree.adj_list[i])])
                for i in range(self.sims.shape[0])
            ])
        self.score(tree.adj_list, root, score_list)
        max_level = np.max([s[1] for s in score_list])
        weights = np.array([s[1] / max_level for s in score_list])
        weights = weights / sum(weights)
        return np.sum([s[0] * w for w, s in zip(weights, score_list)]), root

    def score(
        self,
        adj_list: List[Set[int]],
        root: int,
        scores: List[Tuple[float, int]],
        prev: Optional[int] = None,
        level: int = 1
    ) -> None:
        children = [n for n in adj_list[root] if n != prev]
        if len(children) == 0:
            return

        parent_score = 0
        sibling_score = 0
        int_len_score = 0
        for i in range(len(children)):
            child1 = children[i]
            parent_score += self.sims[root, child1]
            root_words = self.words[root].split(' ')
            child1_words = self.words[child1].split(' ')
            int_len_score += (
                (len(root_words) <= len(child1_words)) and
                (len(set(root_words).intersection(set(child1_words))) > 0)
            )
            for j in range(i + 1, len(children)):
                child2 = children[j]
                sibling_score += self.sims[child1, child2]

        if len(children) > 1:
            parent_score = parent_score / len(children)
            sibling_score = sibling_score / (len(children) * (len(children) - 1) / 2)
            int_len_score = int_len_score / len(children)

        n_children_term = (
            1 if len(children) in range(self.min_children, self.max_children + 1) else -1
        )
        scores.append((
            self.parent_sim_weight * parent_score +
            self.sibling_sim_weight * sibling_score +
            self.n_children_weight * n_children_term +
            self.length_intersection_weight * int_len_score, level)
        )
        for child in children:
            self.score(adj_list, child, scores, root, level+1)


def generate_ontology(
    words: List[str],
    sims: np.array,
    num_generations: int = 500,
    population_size: int = 100,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.1,
    keep_next_gen: int = 20,
    callback: Callable = dummy_callback
) -> Tuple[np.array, int]:
    seq_len = sims.shape[0] - 2
    max_val = sims.shape[0]
    rng = np.random.default_rng(0)
    generation = rng.integers(max_val, size=(population_size, seq_len))
    fitness_function = FitnessFunction(words, sims).fitness
    ticks = iter(np.linspace(0, 1.0, num_generations))

    for _ in range(num_generations):
        callback(next(ticks))
        fitness = np.array([
            fitness_function(Tree.from_prufer_sequence(generation[i, :], words))[0]
            for i in range(population_size)
        ])
        fitness = fitness / fitness.sum()
        new_generation = np.zeros(generation.shape, dtype=int)
        parents_to_keep = np.argsort(fitness)[-keep_next_gen:]
        new_generation[:keep_next_gen, :] = generation[parents_to_keep, :]
        for i in range(keep_next_gen, population_size):
            if rng.uniform() > crossover_probability:
                choice = rng.choice(
                    np.arange(len(fitness)),
                    size=1,
                    p=fitness,
                    replace=False
                )[0]
                new_generation[i, :] = generation[choice, :]
            else:
                u, v = rng.choice(
                    np.arange(len(fitness)),
                    size=2,
                    p=fitness,
                    replace=False
                )
                point = rng.integers(seq_len // 2)
                new_generation[i, :point] = generation[u, :point]
                new_generation[i, point:point+seq_len//2] = generation[v, point:point+seq_len//2]
                new_generation[i, point+seq_len//2:] = generation[u, point+seq_len//2:]
            mutation_idx = np.where(rng.uniform(size=seq_len) < mutation_probability)[0]
            new_generation[i, mutation_idx] = rng.integers(seq_len, size=len(mutation_idx))
        generation = new_generation

    # select the best solution
    scores, roots = list(), list()
    for i in range(population_size):
        score, root = fitness_function(Tree.from_prufer_sequence(generation[i, :], words))
        scores.append(score)
        roots.append(root)
    best = np.argmax(scores)
    return generation[best, :], roots[best]


class OntologyHandler:
    def __init__(self):
        self.embedder = SBERT()

    def generate(
        self,
        words: List[str],
        callback: Callable = dummy_callback
    ) -> Tuple[Dict, int]:
        embeddings = self.embedder(words, wrap_callback(callback, end=0.1))
        non_none = [(w, e) for w, e in zip(words, embeddings) if e is not None]
        skipped = len(words) - len(non_none)
        if len(non_none) == 0:
            return {}, skipped
        words, embeddings = zip(*non_none)
        sims = self._get_similarities(embeddings)
        callback(0.2)
        if len(words) == 1:
            return {words[0]: {}}, skipped
        elif len(words) == 2:
            return {sorted(words)[0]: {sorted(words)[1]: {}}}, skipped
        elif len(words) == 3:
            root = np.argmin(np.sum(sims, axis=1))
            rest = sorted([words[i] for i in range(3) if i != root])
            return {words[root]: {rest[0]: {}, rest[1]: {}}}, skipped
        ontology, root = generate_ontology(
            words,
            sims,
            callback=wrap_callback(callback, start=0.2)
        )
        return Tree.from_prufer_sequence(ontology, words, root).to_dict(), skipped

    def insert_in_tree(
        self, tree: Tree, words: List[str], callback: Callable
    ) -> Tuple[Tree, int]:
        skipped = 0
        for iw, word in enumerate(words, start=1):
            tree.adj_list.append(set())
            tree.labels.append(word)
            embeddings = self.embedder(tree.labels)
            if embeddings[-1] is None:
                # the last embedding is for the newly provided word
                # if embedding is not successful skip it
                skipped += 1
                continue
            sims = self._get_similarities(embeddings)
            idx = len(tree.adj_list) - 1
            fitness_function = FitnessFunction(tree.labels, sims).fitness
            scores = list()
            for i in range(idx):
                tree.adj_list[i].add(idx)
                tree.adj_list[idx].add(i)
                scores.append(fitness_function(tree, tree.root)[0])
                tree.adj_list[i].remove(idx)
                tree.adj_list[idx].remove(i)
            best = np.argmax(scores)
            tree.adj_list[best].add(idx)
            tree.adj_list[idx].add(best)
            callback(iw / len(words))
        return tree, skipped

    __TEMP_ROOT = "<R-O-O-T>"  # root different from any word

    def insert(
        self, tree: Dict, words: List[str], callback: Callable = dummy_callback
    ) -> Tuple[Dict, int]:
        dummy_root_used = False
        if len(tree) > 1:
            # if ontology has multiple roots insert temporary root
            tree = {self.__TEMP_ROOT: tree}
            dummy_root_used = True

        tree = Tree.from_dict(tree)
        tree, skipped = self.insert_in_tree(tree, words, callback)
        tree = tree.to_dict()

        if dummy_root_used:
            # if temporary root inserted remove it
            tree = tree[self.__TEMP_ROOT]
        return tree, skipped

    def score(self, tree: Dict, callback: Callable = dummy_callback) -> float:
        if not tree:
            return 0
        tree = Tree.from_dict(tree)
        embeddings = self.embedder(tree.labels, wrap_callback(callback, end=0.7))
        sims = self._get_similarities(embeddings)
        callback(0.9)
        fitness_function = FitnessFunction(tree.labels, sims).fitness
        return fitness_function(tree, tree.root)[0]

    @staticmethod
    def _get_similarities(embeddings: np.array) -> np.array:
        return cosine_similarity(embeddings, embeddings)
