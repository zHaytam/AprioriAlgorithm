import pandas as pd
from itertools import combinations, chain


class Apriori:
    """
    Represents an apriori algorithm instance.
    """

    def __init__(self, df, min_support, min_confidence):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._construct_transactions(df)
        self.frequent_itemsets = None
        self.association_rules = None

    def _construct_transactions(self, df):
        """
        Generates all the transactions based on the column names and the values.
        """

        self.transactions = []
        self.one_item_sets = set()
        cols = df.columns.values

        for row in df.values:
            transaction = frozenset(['%s_%s' % (col_, row_) for row_, col_ in zip(row, cols)])
            self.transactions.append(transaction)
            for item in transaction:
                self.one_item_sets.add(frozenset([item]))

    @staticmethod
    def _generate_candidates(itemsets, k):
        """
        :return: All the unique combinations (candidates) of length k.
        """

        candidates = [x.union(y) for x in itemsets for y in itemsets]
        return set([candidate for candidate in candidates if len(candidate) == k])

    def generate_frequent_itemsets(self):
        """
        Generates all the frequent itemsets.
        """

        self.frequent_itemsets = dict()
        itemsets = self._frequent_itemsets_supports(self.one_item_sets)
        self.frequent_itemsets.update(itemsets)

        k = 2
        while True:
            candidates = self._generate_candidates(itemsets.keys(), k)
            itemsets = self._frequent_itemsets_supports(candidates)

            # No more frequent itemsets
            if not itemsets:
                break

            self.frequent_itemsets.update(itemsets)
            k += 1

    def _frequent_itemsets_supports(self, itemsets):
        """
        Generates all the (itemset, support) of all the itemsets.
        """

        transactions_len = len(self.transactions)
        itemsets_supports = dict()

        for item in itemsets:
            support = float(sum(1 for row in self.transactions if item.issubset(row))) / transactions_len
            if support >= self.min_support:
                itemsets_supports[item] = support

        return itemsets_supports

    @staticmethod
    def _generate_subsets(itemset):
        """
        :return: All the subsets of the itemset.
        """

        combs = [combinations(itemset, i) for i in range(1, len(itemset))]
        return [frozenset(x) for x in chain(*combs)]

    def generate_association_rules(self, itemsets_len=None):
        """
        Generates all the association rules using the frequent itemsets.
        """

        if not self.frequent_itemsets:
            raise ValueError('Please generate the frequent itemsets first.')

        self.association_rules = []
        for itemset, support in self.frequent_itemsets.items():

            # Length checks
            if len(itemset) < 2:
                continue

            if itemsets_len and len(itemset) != itemsets_len:
                continue

            for X in self._generate_subsets(itemset):
                Y = itemset.difference(X)

                # This will always be valid because of the property of apriori
                confidence = support / self.frequent_itemsets[X]
                if confidence >= self.min_confidence:
                    self.association_rules.append((X, Y, confidence))


def get_test_data():
    """
    :return: A sample dataframe (titanic).
    """

    df = pd.read_csv('titanic_train.csv')
    return df[['Pclass', 'Sex', 'Age', 'Survived']]


if __name__ == '__main__':
    test_df = get_test_data()
    apriori = Apriori(test_df, 0.1, 0.5)
    print('Constructed transactions and 1-itemsets.')
    print(apriori.transactions[0])
    print(len(apriori.one_item_sets))
    apriori.generate_frequent_itemsets()
    print(apriori.frequent_itemsets)

    # frequent 3-itemsets
    print([(fi, apriori.frequent_itemsets[fi]) for fi in apriori.frequent_itemsets if len(fi) == 3])

    apriori.generate_association_rules()
    print(apriori.association_rules)

