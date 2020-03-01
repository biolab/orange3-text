from Orange.widgets.settings import PerfectDomainContextHandler


class AlmostPerfectContextHandler(PerfectDomainContextHandler):
    """
    This context compares both domains and demands that both domain matches
    in share_domain_matches (e.g. 0.9) of variables. The position of variables
    (attribute, meta, class_var) is not important since widget that use this
    handler do not use their values directly.

    Attributes
    ----------
    share_domain_matches
        The share of domain attributes that need to match.
    """
    def __init__(self, share_domain_matches: float) -> None:
        super().__init__()
        self.share_domain_matches = share_domain_matches

    def match(self, context, domain, attributes, class_vars, metas):
        context_vars = context.attributes + context.class_vars + context.metas
        domain_vars = attributes + class_vars + metas
        matching_vars = [var for var in context_vars if var in domain_vars]

        return (self.PERFECT_MATCH
                if (len(matching_vars) / len(domain_vars)
                    > self.share_domain_matches)
                else self.NO_MATCH)
