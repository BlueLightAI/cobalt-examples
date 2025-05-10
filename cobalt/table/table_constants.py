DEFAULT_NUMERIC_FORMAT = "{:.4g}"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

NUMBER_OF_DF_ROWS_TO_DISPLAY = 2000
CONTAINS = "contains"
IS = "is"
EQUALS = "eq"

DEFAULT_OPERATOR_SELECT_ITEMS = [
    {
        "text": "contains",
        "value": "contains",
    },
]
MANUAL_TEXT_SELECT_ITEMS = [
    {
        "header": "CATEGORICAL",
    },
    {
        "text": "is",
        "value": "is",
    },
]
MANUAL_NUMERIC_SELECT_ITEMS = [
    {
        "header": "CATEGORICAL",
    },
    {
        "text": "is",
        "value": "eq",
    },
]
NUMERICAL_OPERATOR_SELECT_ITEMS = [
    {
        "header": "NUMERICAL",
    },
    {
        "text": "equals",
        "value": "eq",
    },
    {
        "text": "more than",
        "value": "gt",
    },
    {
        "text": "less than",
        "value": "lt",
    },
    {
        "text": "more than or equal to",
        "value": "gte",
    },
    {
        "text": "less than or equal to",
        "value": "lte",
    },
]
CATEGORICAL_TEXT_OPERATOR_SELECT_ITEMS = [
    {
        "header": "CATEGORICAL",
    },
    {
        "text": "is",
        "value": "is",
    },
    {
        "text": "contains",
        "value": "contains",
    },
]

OPERATOR_MAP = {
    "is_case_sensitive_on": IS,
    "is_case_sensitive_off": CONTAINS,
    "contains_sensitive_on": CONTAINS,
    "contains_sensitive_off": CONTAINS,
    "eq": "=",
    "gt": ">",
    "lt": "<",
    "gte": ">=",
    "lte": "<=",
}
