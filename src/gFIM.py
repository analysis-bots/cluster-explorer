#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of algorithms related to itemsets.
"""

import itertools
import numbers
import typing
from typing import Set
import collections
from dataclasses import field, dataclass
import collections.abc
# import multiprocessing
from multiprocessing import Pool, cpu_count
import concurrent.futures
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from black.trans import defaultdict
from multiset import Multiset


@dataclass
class ItemsetCount:
    itemset_count: int = 0
    members: set = field(default_factory=set)


class TransactionManager:
    # The brilliant transaction manager idea is due to:
    # https://github.com/ymoch/apyori/blob/master/apyori.py

    def __init__(self, transactions: typing.Iterable[typing.Iterable[typing.Hashable]], item_ancestors_dict):
        # A lookup that returns indices of transactions for each item
        self.indices_by_item = collections.defaultdict(set)

        # Populate
        i = -1
        for i, transaction in enumerate(transactions):
            for item in transaction:
                self.indices_by_item[item].add(i)
                if item in item_ancestors_dict:
                    for ancestor in item_ancestors_dict[item]:
                        self.indices_by_item[(item[0], ancestor)].add(i)

        # Total number of transactions
        self._transactions = i + 1

    @property
    def items(self):
        return set(self.indices_by_item.keys())

    def __len__(self):
        return self._transactions

    def transaction_indices(self, transaction: typing.Iterable[typing.Hashable]):
        """Return the indices of the transaction."""

        transaction = set(transaction)  # Copy
        item = transaction.pop()
        indices = self.indices_by_item[item]
        while transaction:
            item = transaction.pop()
            indices = indices.intersection(self.indices_by_item[item])
        return indices

    def transaction_indices_sc(self, transaction: typing.Iterable[typing.Hashable], min_support: float = 0):
        """Return the indices of the transaction, with short-circuiting.

        Returns (over_or_equal_to_min_support, set_of_indices)
        """

        # Sort items by number of transaction rows the item appears in,
        # starting with the item beloning to the most transactions
        transaction = sorted(transaction, key=lambda item: len(self.indices_by_item[item]), reverse=True)

        # Pop item appearing in the fewest
        item = transaction.pop()
        indices = self.indices_by_item[item]
        support = len(indices) / len(self)
        if support < min_support:
            return False, None

        # The support is a non-increasing function
        # Sorting by number of transactions the items appear in is a heuristic
        # to make the support drop as quickly as possible
        while transaction:
            item = transaction.pop()
            indices = indices.intersection(self.indices_by_item[item])
            support = len(indices) / len(self)
            if support < min_support:
                return False, None

        # No short circuit happened
        return True, indices

    def remove_non_candidates(self, found_itemsets: typing.Dict[tuple, int]):
        """Remove items that are not part of any candidate itemsets in found_itemsets."""
        # Extract itemsets from the keys of the found_itemsets dictionary
        found_itemsets_keys = set(itertools.chain.from_iterable(found_itemsets.keys()))

        # Determine which items to remove: those not in the found_itemsets
        items_to_remove = set(self.indices_by_item.keys()) - found_itemsets_keys
        for item in items_to_remove:
            del self.indices_by_item[item]


def join_step(itemsets: typing.List[tuple]):
    """
    Join k length itemsets into k + 1 length itemsets.

    This algorithm assumes that the list of itemsets are sorted, and that the
    itemsets themselves are sorted tuples. Instead of always enumerating all
    n^2 combinations, the algorithm only has n^2 runtime for each block of
    itemsets with the first k - 1 items equal.

    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k, to be joined to k + 1 length
        itemsets.

    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> list(join_step(itemsets))
    [(1, 2, 3, 4), (1, 3, 4, 5)]
    """
    i = 0
    # Iterate over every itemset in the itemsets
    while i < len(itemsets):
        # The number of rows to skip in the while-loop, initially set to 1
        skip = 1

        # Get all but the last item in the itemset, and the last item
        *itemset_first, itemset_last = itemsets[i]

        # We now iterate over every itemset following this one, stopping
        # if the first k - 1 items are not equal. If we're at (1, 2, 3),
        # we'll consider (1, 2, 4) and (1, 2, 7), but not (1, 3, 1)

        # Keep a list of all last elements, i.e. tail elements, to perform
        # 2-combinations on later on
        tail_items = [itemset_last]
        tail_items_append = tail_items.append  # Micro-optimization

        # Iterate over ever itemset following this itemset
        for j in range(i + 1, len(itemsets)):
            # Get all but the last item in the itemset, and the last item
            *itemset_n_first, itemset_n_last = itemsets[j]

            # If it's the same, append and skip this itemset in while-loop
            if itemset_first == itemset_n_first:
                # Micro-optimization
                tail_items_append(itemset_n_last)
                skip += 1

            # If it's not the same, break out of the for-loop
            else:
                break

        # For every 2-combination in the tail items, yield a new candidate
        # itemset, which is sorted.
        itemset_first_tuple = tuple(itemset_first)
        # itemsets_list = sorted(item for item in large_itemsets[k - 1].keys())
        for a, b in itertools.combinations(tail_items, 2):
            if a[0] != b[0]:
                yield itemset_first_tuple + (a,) + (b,)

        # Increment the while-loop counter
        i += skip


def prune_step(itemsets: typing.Iterable[tuple], possible_itemsets: typing.List[tuple]):
    """
    Prune possible itemsets whose subsets are not in the list of itemsets.

    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k.
    possible_itemsets : list of itemsets
        A list of possible itemsets of length k + 1 to be pruned.

    Examples
    -------
    >>> itemsets = [('a', 'b', 'c'), ('a', 'b', 'd'),
    ...             ('b', 'c', 'd'), ('a', 'c', 'd')]
    >>> possible_itemsets = list(join_step(itemsets))
    >>> list(prune_step(itemsets, possible_itemsets))
    [('a', 'b', 'c', 'd')]
    """

    # For faster lookups
    itemsets = set(itemsets)

    # Go through every possible itemset
    for possible_itemset in possible_itemsets:
        # Remove 1 from the combination, same as k-1 combinations
        # The itemsets created by removing the last two items in the possible
        # itemsets must be part of the itemsets by definition,
        # due to the way the `join_step` function merges the sorted itemsets

        for i in range(len(possible_itemset) - 2):
            removed = possible_itemset[:i] + possible_itemset[i + 1:]

            # If every k combination exists in the set of itemsets,
            # yield the possible itemset. If it does not exist, then it's
            # support cannot be large enough, since supp(A) >= supp(AB) for
            # all B, and if supp(S) is large enough, then supp(s) must be large
            # enough for every s which is a subset of S.
            # This is the downward-closure property of the support function.
            if removed not in itemsets:
                break

        # If we have not breaked yet
        else:
            yield possible_itemset


def apriori_gen(itemsets: typing.List[tuple]):
    """
    Compute all possible k + 1 length supersets from k length itemsets.

    This is done efficiently by using the downward-closure property of the
    support function, which states that if support(S) > k, then support(s) > k
    for every subset s of S.

    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k.

    Examples
    -------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> possible_itemsets = list(join_step(itemsets))
    >>> list(prune_step(itemsets, possible_itemsets))
    [(1, 2, 3, 4)]
    """
    possible_extensions = join_step(itemsets)
    yield from prune_step(itemsets, possible_extensions)


def process_candidate(candidate, manager, min_support):
    over_min_support, indices = manager.transaction_indices_sc(candidate, min_support=min_support)
    if over_min_support:
        return candidate, len(indices)
    return candidate, 0


def check_support(candidate, manager, min_support):
    over_min_support, indices = manager.transaction_indices_sc(candidate, min_support=min_support)
    return (candidate, len(indices)) if over_min_support else None


def itemsets_from_transactions(
        transactions: typing.Iterable[typing.Union[set, tuple, list]],
        item_ancestors_dict: dict,
        min_support: float,
        max_length: int = 8,
        verbosity: int = 0,
        output_transaction_ids: bool = False,
        mode: str = 'conjunction',
):
    if mode == 'conjunction':
        return apriori(transactions, item_ancestors_dict, min_support, max_length, verbosity, output_transaction_ids)
    elif mode == 'disjunction':
        return dssrm(transactions, item_ancestors_dict, min_support, max_length, verbosity, output_transaction_ids)
    else:
        raise ValueError("Invalid mode. Choose either 'conjunction' or 'disjunction'.")


def apriori(
        transactions: typing.Iterable[typing.Union[set, tuple, list]],
        item_ancestors_dict: dict,
        min_support: float,
        max_length: int = 8,
        verbosity: int = 0,
        output_transaction_ids: bool = False,
):
    """
    Compute itemsets from transactions by building the itemsets bottom up and
    iterating over the transactions to compute the support repedately. This is
    the heart of the Apriori algorithm by Agrawal et al. in the 1994 paper.

    Parameters
    ----------
    transactions : a list of itemsets (tuples/sets/lists with hashable entries)
    min_support : float
        The minimum support of the itemsets, i.e. the minimum frequency as a
        percentage.
    max_length : int
        The maximum length of the itemsets.
    verbosity : int
        The level of detail printing when the algorithm runs. Either 0, 1 or 2.
    output_transaction_ids : bool
        If set to true, the output contains the ids of transactions that
        contain a frequent itemset. The ids are the enumeration of the
        transactions in the sequence they appear.

    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> transactions = [(1, 3, 4), (2, 3, 5), (1, 2, 3, 5), (2, 5)]
    >>> itemsets, _ = itemsets_from_transactions(transactions, min_support=2/5)
    >>> itemsets[1] == {(1,): 2, (2,): 3, (3,): 3, (5,): 3}
    True
    >>> itemsets[2] == {(1, 3): 2, (2, 3): 2, (2, 5): 3, (3, 5): 2}
    True
    >>> itemsets[3] == {(2, 3, 5): 2}
    True
    """

    # STEP 0 - Sanitize user inputs
    # -----------------------------
    if not (isinstance(min_support, numbers.Number) and (0 <= min_support <= 1)):
        raise ValueError("`min_support` must be a number between 0 and 1.")

    # Store in transaction manager
    manager = TransactionManager(transactions, item_ancestors_dict)

    # If no transactions are present
    transaction_count = len(manager)
    if transaction_count == 0:
        return dict(), 0  # large_itemsets, num_transactions

    # STEP 1 - Generate all large itemsets of size 1
    # ----------------------------------------------
    if verbosity > 0:
        print("Generating itemsets.")
        print(" Counting itemsets of length 1.")

    candidates: typing.Dict[tuple, int] = {(item,): len(indices) for item, indices in manager.indices_by_item.items()}
    large_itemsets: typing.Dict[int, typing.Dict[tuple, int]] = {
        1: {item: count for (item, count) in candidates.items() if (count / len(manager)) >= min_support}
    }

    if verbosity > 0:
        print("  Found {} candidate itemsets of length 1.".format(len(manager.items)))
        print("  Found {} large itemsets of length 1.".format(len(large_itemsets.get(1, dict()))))
    if verbosity > 1:
        print("    {}".format(list(item for item in large_itemsets.get(1, dict()).keys())))

    # If large itemsets were found, convert to dictionary
    if not large_itemsets.get(1, dict()):
        return dict(), 0  # large_itemsets, num_transactions

    # STEP 2 - Build up the size of the itemsets
    # ------------------------------------------

    # While there are itemsets of the previous size
    k = 2
    while large_itemsets[k - 1] and (max_length != 1):
        if verbosity > 0:
            print(" Counting itemsets of length {}.".format(k))

        # STEP 2a) - Build up candidate of larger itemsets

        # Retrieve the itemsets of the previous size, i.e. of size k - 1
        # They must be sorted to maintain the invariant when joining/pruning
        # itemsets_list = sorted(item for item in large_itemsets[k - 1].keys())
        itemsets_list = [item for item in large_itemsets[k - 1].keys()]

        # Gen candidates of length k + 1 by joining, prune, and copy as set
        # This algorithm assumes that the list of itemsets are sorted,
        # and that the itemsets themselves are sorted tuples
        C_k: typing.List[tuple] = list(apriori_gen(itemsets_list))

        if verbosity > 0:
            print("  Found {} candidate itemsets of length {}.".format(len(C_k), k))
        if verbosity > 1:
            print("   {}".format(C_k))

        # If no candidate itemsets were found, break out of the loop
        if not C_k:
            break

        # Prepare counts of candidate itemsets (from the prune step)
        if verbosity > 1:
            print("    Iterating over transactions.")

        # Keep only large transactions
        found_itemsets: typing.Dict[tuple, int] = dict()
        count1 = 0
        # for candidate in C_k:
        #     over_min_support, indices = manager.transaction_indices_sc(candidate, min_support=min_support)
        #     if over_min_support:
        #         found_itemsets[candidate] = len(indices)
        print(f"{count1},{len(C_k)}")
        #     count1+=1

        args_for_parallel = [(candidate, manager, min_support) for candidate in C_k]

        # Use concurrent.futures for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_candidate = {executor.submit(check_support, *args): args[0] for args in args_for_parallel}
            for future in concurrent.futures.as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                result = future.result()
                if result:
                    found_itemsets[candidate] = result[1]

        # ... (rest of your function logic)

        # If no itemsets were found, break out of the loop
        if not found_itemsets:
            break

        # Candidate itemsets were found, add them
        large_itemsets[k] = {i: counts for (i, counts) in found_itemsets.items()}

        if verbosity > 0:
            num_found = len(large_itemsets[k])
            print("  Found {} large itemsets of length {}.".format(num_found, k))
        if verbosity > 1:
            print("   {}".format(list(large_itemsets[k].keys())))
        k += 1

        manager.remove_non_candidates(found_itemsets)

        # Break out if we are about to consider larger itemsets than the max
        if k > max_length:
            break

    if verbosity > 0:
        print("Itemset generation terminated.\n")

    print("endd")
    if output_transaction_ids:
        itemsets_out = {
            length: {
                item: ItemsetCount(itemset_count=count, members=manager.transaction_indices(set(item)))
                for (item, count) in itemsets.items()
            }
            for (length, itemsets) in large_itemsets.items()
        }
        return itemsets_out, len(manager)
    print("endd")

    # manager.remove_non_candidates(C_k)

    return large_itemsets, len(manager)


@dataclass
class Candidate:
    """
    A candidate itemset in the DSSRM algorithm.
    """
    disj_support: float
    conj_support: float
    # We use a frozenset to ensure that the itemset is hashable.
    itemset: frozenset
    itemset_indices_len: float
    closure: set
    complement_closure: set

    def __eq__(self, other):
        return self.itemset == other.itemset

    def __hash__(self):
        return hash(self.itemset)


@dataclass
class CandidateHistory:
    """
    History of candidates in the DSSRM algorithm.
    Used to quickly look up candidates by their itemset, to avoid recomputing closures.
    """
    candidates: set

    def __getitem__(self, key):
        if not isinstance(key, frozenset):
            key = frozenset(key)
        # If the key is a set, return the candidate with the itemset equal to the key
        return next(candidate for candidate in self.candidates if candidate.itemset == key)


class LookupTransactionsTable:
    """
    A lookup table for transactions.
    Used to quickly look up transactions that contain a certain itemset, or do not contain a certain itemset.
    """

    def __init__(self, transactions):
        # Get all of the unique items in the transactions
        keys = set(itertools.chain.from_iterable(transactions))
        self.transaction_dict = defaultdict(set)
        for key in keys:
            self.transaction_dict[key] = set()
        # Populate the transaction dictionary, so that we can quickly look up transactions that contain an item
        for transaction in transactions:
            for key in transaction:
                self.transaction_dict[key].add(transaction)
        # Store all transactions in a set, for quick set operations
        self._all_transactions = set(transactions)

    def __getitem__(self, key):
        """
        Get all transactions that have a non-empty intersection with the key.
        """
        if not isinstance(key, tuple):
            key = tuple(key)
        ret_values = set()
        for item in key:
            ret_values.update(self.transaction_dict[item])
        return ret_values

    def get_non_subsets(self, key):
        """
        Get all transactions that have an empty intersection with the key.
        """
        if not isinstance(key, tuple):
            key = tuple(key)
        # Assuming len(key) << len(transactions), doing the inverse of getitem is faster than iterating over all transactions
        containing_transactions = self[key]
        return self._all_transactions - containing_transactions


def dssrm_prune_step(candidates_i, L_i, candidate_history) -> Set[Candidate]:
    """
    The prune step of the DSSRM algorithm.
    Prunes the candidates by removing those that don't fulfill the 2 conditions:
    1. All subsets of the candidate are in L_i
    2. The candidate is not in the closure of any of its subsets

    :param candidates_i: The candidates to prune
    :param L_i: The large itemsets of length i
    :param candidate_history: The history of candidates

    :return: The pruned candidates
    """
    pruned_candidates = set()
    for candidate in candidates_i:
        # Convert the candidate itemset to a candidate object
        candidate = Candidate(disj_support=0, conj_support=0, itemset=frozenset(candidate), closure=set(),
                              complement_closure=set(), itemset_indices_len=len(candidate))
        # Get all subsets of the candidate and look them up in the candidate history
        all_subsets = set(itertools.chain.from_iterable(
            itertools.combinations(candidate.itemset, r) for r in range(1, len(candidate.itemset))))
        all_subsets = [candidate_history[subset] for subset in all_subsets]
        # If all subsets are in L_i and the candidate is not in the closure of any of its subsets, add it to the pruned candidates
        if all(subset in L_i for subset in all_subsets) and not any(
                candidate.itemset.issubset(subset.closure) for subset in all_subsets):
            pruned_candidates.add(candidate)

    return pruned_candidates


def dssrm(transactions: typing.Iterable[typing.Union[set, tuple, list]],
          item_ancestors_dict: dict,
          min_support: float,
          max_length: int = 8,
          *args, **kwargs) -> typing.Tuple[dict, int]:
    """
    Compute the disjunctive frequent itemsets using the DSSRM algorithm.
    The DSSRM algorithm is presented in the paper "Optimized Mining of a Concise Representation for Frequent Patterns
    Based on Disjunctions Rather than Conjunctions" By Hamrouni et al. (2010).

    :param transactions: The transactions to mine
    :param item_ancestors_dict: A dictionary containing the ancestors of each item
    :param min_support: The minimum support of the itemsets, i.e. the minimum frequency as a percentage.
    :param max_length: The maximum length of the itemsets.

    :return: A dictionary containing the frequent itemsets and the number of transactions
    """
    # Sanitize user inputs
    if not (isinstance(min_support, numbers.Number) and (0 <= min_support <= 1)):
        raise ValueError("`min_support` must be a number between 0 and 1.")

    # Store in transaction manager
    manager = TransactionManager(transactions, item_ancestors_dict)

    # If no transactions are present
    transaction_count = len(manager)
    if transaction_count == 0:
        return dict(), 0  # large_itemsets, num_transactions

    # DSSRM considers the support as the number of transactions containing the itemset, not the frequency
    min_support = int(min_support * transaction_count)

    # EDCP - Essential Disjunctive Closed Patterns
    edcp = set()
    # FEP - Frequent Essential Patterns
    fep = set()
    # Candidate itemsets of length 1. All items are candidates at this stage.
    candidates_i = [Candidate(disj_support=0, conj_support=0, itemset=frozenset({item}),
                              closure=set(), complement_closure=set(),
                              itemset_indices_len=len(indices))
                    for item, indices in manager.indices_by_item.items()]
    history = CandidateHistory(candidates=set(candidates_i))
    items = set(manager.items)
    # Replace values in the transactions with their ancestors
    for j, transaction in enumerate(transactions):
        for i, item in enumerate(transaction):
            if item in item_ancestors_dict:
                # We use frozensets to ensure that the itemset is hashable and thus quickly looked up
                transaction[i] = frozenset({(item[0], ancestor) for ancestor in item_ancestors_dict[item]})
        transactions[j] = frozenset(itertools.chain.from_iterable(transaction))

    transactions_lookup = LookupTransactionsTable(transactions=transactions)
    # Create a single large multi-set of all transactions.
    # Computing the supports, in the paper, is done by iterating over the transactions.
    # Using a multi-set allows us to avoid iterating over the transactions, getting a massive speedup (approx x3 as fast).
    transactions = Multiset(itertools.chain.from_iterable(transactions))

    next_pattern_len = 1

    num_cores = cpu_count()

    # Main loop - while there are candidates to consider and the pattern length is less than or equal to the max length
    while len(candidates_i) > 0 and next_pattern_len <= max_length:
        candidates_i = list(candidates_i)
        candidates_chunks = [candidates_i[i::num_cores] for i in range(num_cores)]

        # Compute the supports and closures of the candidates in parallel.
        L_i = set()
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            futures = [
                executor.submit(compute_supports_closures, transactions, min_support, candidates_chunk, edcp, items,
                                transactions_lookup)
                for candidates_chunk in candidates_chunks]
            for future in as_completed(futures):
                L_i = L_i.union(future.result())

        # Add the candidates to the Frequent Essential Patterns
        fep = fep.union({(candidate.itemset, candidate.disj_support) for candidate in L_i})
        # Generate and prune the next candidates. These are passed as a list of tuples to better match the expected input,
        # but it should work just as well as a list of frozensets.
        candidates_i = list(apriori_gen([tuple(candidate.itemset) for candidate in L_i]))
        candidates_i = dssrm_prune_step(candidates_i, L_i, history)
        # Save the candidates in the history and increment the pattern length
        history.candidates = history.candidates.union(candidates_i)
        next_pattern_len += 1

    # edcp contains the closures of the frequent essential patterns.
    # In the paper, they show that the full representation of the disjunctive patterns is given by the union of
    # edcp and fep. However, the closures can be massive, so we only take the closures that are of length <= max_length.
    # This may leave edcp empty, but we still get some patterns via fep.
    edcp = {pattern[0] for pattern in edcp if len(pattern[0]) <= max_length}
    fep = fep.union(edcp)

    # Convert the frequent essential patterns to the expected output format
    return_dict = {}
    for pattern in fep:
        pattern_len = len(pattern[0])
        if pattern_len not in return_dict:
            return_dict[pattern_len] = {}
        # Convert from a tuple with a frozenset in it in the 0 index to a tuple
        pattern_items = tuple(pattern[0])
        # The actual value is irrelevant, we only care about the key, but we need it for the expected output
        return_dict[pattern_len][pattern_items] = 1

    return return_dict, len(manager)


def compute_supports_closures(transactions, min_supp, candidates, edcp, items, lookup_table) -> Set[Candidate]:
    """
    The support computation step of the DSSRM algorithm.
    Computes the closure, disjunctive support, and conjunctive support of the candidates, keeping only
    those that have a support greater than or equal to the minimum support.

    :param transactions: The transactions to mine
    :param min_supp: The minimum support of the itemsets, i.e. the minimum frequency as a percentage.
    :param candidates: The candidates to compute the support of
    :param edcp: The set of Essential Disjunctive Closed Patterns
    :param items: The set of all items
    :param lookup_table: A lookup table for the transactions, allowing for quick lookups of transactions containing or not containing an itemset

    :return: The large itemsets of length i
    """
    L_i = set()
    # The original algorithm iterates over the transactions and the candidates, but we can get a massive speedup
    # by using a multi-set of all transactions instead.
    # The iterations are replaced by set operations, which are much faster.
    for candidate in candidates:
        # Omega is the intersection of the candidate itemset and the transaction. This is the name used in the paper.
        omega = transactions.intersection(candidate.itemset)
        # Any transaction that does not, in any way, intersect with the candidate is a subset of the complement of the candidate's closure
        # Thus, we add all of the items from those transactions to the complement closure
        candidate.complement_closure = candidate.complement_closure.union(
            itertools.chain.from_iterable(lookup_table.get_non_subsets(candidate.itemset))
        )

        if len(omega) != 0:
            multiplicities = [transactions.get(item, 0) for item in omega]
            # The disjunctive support is updated to the maximum number of occurrences of any item in the intersection
            candidate.disj_support += max(multiplicities)
            if len(omega) == len(candidate.itemset):
                # The conjunctive support is updated to the minimum number of occurrences of any item in the intersection
                # Aka the number of transactions containing the candidate in full
                candidate.conj_support += min(multiplicities)

    # For each candidate, if its conjunctive support is greater than or equal to the minimum support, add it to L_i.
    # Compute the closure of the candidate and add it to edcp as well.
    for candidate in candidates:
        if candidate.conj_support >= min_supp:
            L_i.add(candidate)
            candidate.closure = items - candidate.complement_closure
            edcp.add((frozenset(candidate.closure), candidate.disj_support))

    return L_i


if __name__ == "__main__":
    import pytest

    print("aaaa")
    pytest.main(args=[".", "--doctest-modules", "-v"])
