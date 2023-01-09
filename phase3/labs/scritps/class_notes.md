Class for EDA:

## Preprocessing Class (Handle_Bars)
    ### Information
        def column names by type
            -numerical list
            -categorical list
            -int/float/object
        def __repr__
            -maybe have arguments for what you want to print
                -shape
                -types
                -Table of contents
                -level of verbosity
        def graph
            - pipelines for types of graphs
                -hex mapping graph
                -histogram
                    -outputs standard for numerical data, but does value_counts for categorical
                    -can give the second order distributions
                -test graphing
    ### Procession/Manipulation
        def encoding
            -encoding categorical data
        def mapping
            -mapping
            -column-wise functions
        def bridge
            -algorithms for joining new columns
        def cleaning
            -just adding cleaning functionality as you need it

## Model Class (Data_Loom)
    ### Prep
        - Splitting
        - Stadardization
        - feature selector
    ### Fitting
        - fitting, predicting
        - cross validation
    ### Testing
        - Automate Splitting, fitting and feature selection
        - Regularization
