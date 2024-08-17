from src.Dashboard.view import (
    OptionFilters,
    UseCases,
    Dashboard, 
    AboutMetaMP,
    WelcomePage,
    DashboardMap, 
    DashboardOthers,
    SummaryStatistics,  
    MembraneProteinList, 
    SearchMergedDatabases,
    AttributeVisualization,
    DashboardInconsistencies
)
from src.Filters.view import (
    Filters, 
    GraphOptions, 
    MissingFilterKit, 
    allowMissingPerc,
    normalizationOptions, 
    dataSplitPercOptions,
    PCAComponentsOptions,
    MachineLearningOptions, 
    trainAndTestSplitOptions,
    dimensionalityReductionOptions
)

def routes(api):
    api.add_resource(Filters, '/filters')
    api.add_resource(Dashboard, '/dashboard')
    api.add_resource(UseCases, '/get-use-cases')
    api.add_resource(WelcomePage, '/welcome-page')
    api.add_resource(AboutMetaMP, '/about-metamp')
    api.add_resource(DashboardMap, '/dashboard-map')
    api.add_resource(OptionFilters, '/option-filters')
    api.add_resource(MembraneProteinList, '/data-list')
    api.add_resource(GraphOptions, '/graph-options-kit')
    api.add_resource(DashboardOthers, '/dashboard-others')
    api.add_resource(MissingFilterKit, '/missing-filter-kit')
    api.add_resource(allowMissingPerc, '/allow-missing-perc-kit')
    api.add_resource(SearchMergedDatabases, '/search-merged-db')
    api.add_resource(SummaryStatistics, '/get-summary-statistics')
    api.add_resource(AttributeVisualization, '/venn-attribute-desc')
    api.add_resource(normalizationOptions, '/normalization-options-kit')
    api.add_resource(PCAComponentsOptions, '/PCA-components-options-kit')
    api.add_resource(dataSplitPercOptions, '/data-split-perc-options-kit')
    api.add_resource(DashboardInconsistencies, '/dashboard-inconsistencies')
    api.add_resource(MachineLearningOptions, '/machine-learning-options-kit')
    api.add_resource(trainAndTestSplitOptions, '/train-and-test-split-options-kit')
    api.add_resource(dimensionalityReductionOptions, '/dimensionality-reduction-options-kit')