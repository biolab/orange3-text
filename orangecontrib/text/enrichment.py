from orangecontrib.text.stats import false_discovery_rate, hypergeom_p_values


def enrichment(data, selected_data, callback=None):
    """
        Computes p-values and false discovery rate for a subset.
        Method adapted from
        https://en.wikipedia.org/wiki/Gene_set_enrichment_analysis.

        Args:
            data (numpy.array): all examples in rows, theirs features in columns.
            selected_data (numpy.array): selected examples in rows, theirs
            features in columns.
            callback: callback function used for printing progress of
            hypergeom_p_values

        Returns: list of features, p-values and fdr values

        """
    if data.domain != selected_data.domain:
        raise AttributeError("Domains do not match.")

    features = [i.name for i in selected_data.domain.attributes]
    p_values = hypergeom_p_values(data.X, selected_data.X, callback=callback)
    fdr_values = false_discovery_rate(p_values)
    return features, p_values, fdr_values
