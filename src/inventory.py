import numpy as np
import pandas as pd


Z_VALUES = {
    0.90: 1.282,
    0.95: 1.645,
    0.975: 1.960,
    0.99: 2.326,
}


def get_z_value(service_level):
    """
    Return the z-score corresponding to the desired
    service level.
    """

    service_level = float(service_level)

    if service_level in Z_VALUES:
        return Z_VALUES[service_level]

    # Fallback to the standard 95% service level.
    return 1.645


def clean_forecast(predictions):
    """
    Clean model predictions.

    Forecast demand cannot be negative, NaN, or infinite.
    """

    predictions = np.asarray(predictions, dtype=float)

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return np.maximum(predictions, 0.0)


def calculate_demand_statistics(historical_sales):
    """
    Calculate demand variability from historical sales.

    historical_sales:
        2D array where each row represents one
        product/store observation and columns represent
        historical demand periods.
    """

    historical_sales = np.asarray(
        historical_sales,
        dtype=float
    )

    if historical_sales.ndim != 2:
        raise ValueError(
            "historical_sales must be a 2D array."
        )

    demand_mean = np.mean(
        historical_sales,
        axis=1
    )

    demand_std = np.std(
        historical_sales,
        axis=1
    )

    return demand_mean, demand_std


def calculate_safety_stock(
    demand_std,
    lead_time=1,
    service_level=0.95,
):
    """
    Calculate safety stock.

    Formula:

        Safety Stock = Z × demand_std × sqrt(lead_time)
    """

    demand_std = np.asarray(
        demand_std,
        dtype=float
    )

    lead_time = max(
        float(lead_time),
        1.0
    )

    z = get_z_value(service_level)

    safety_stock = (
        z
        * demand_std
        * np.sqrt(lead_time)
    )

    return np.maximum(
        safety_stock,
        0.0
    )


def calculate_reorder_point(
    forecast_demand,
    safety_stock,
    lead_time=1,
):
    """
    Calculate reorder point.

    Formula:

        Reorder Point =
            Forecast demand × Lead Time
            + Safety Stock
    """

    forecast_demand = np.asarray(
        forecast_demand,
        dtype=float
    )

    safety_stock = np.asarray(
        safety_stock,
        dtype=float
    )

    lead_time = max(
        float(lead_time),
        1.0
    )

    reorder_point = (
        forecast_demand * lead_time
        + safety_stock
    )

    return np.maximum(
        reorder_point,
        0.0
    )


def calculate_order_quantity(
    reorder_point,
    current_stock,
):
    """
    Calculate recommended order quantity.

    IMPORTANT:
    current_stock must be an actual stock value.
    We do NOT use the last sales observation as stock.
    """

    reorder_point = np.asarray(
        reorder_point,
        dtype=float
    )

    current_stock = np.asarray(
        current_stock,
        dtype=float
    )

    current_stock = np.maximum(
        current_stock,
        0.0
    )

    order_quantity = np.maximum(
        reorder_point - current_stock,
        0.0
    )

    return np.ceil(
        order_quantity
    ).astype(int)


def create_inventory_plan(
    predictions,
    historical_sales,
    current_stock,
    lead_time=1,
    service_level=0.95,
):
    """
    Create an inventory decision plan.

    Parameters
    ----------
    predictions:
        Forecast demand for the next period.

    historical_sales:
        Historical demand for each product/store.

    current_stock:
        Actual available inventory.

    lead_time:
        Number of demand periods required for replenishment.

    service_level:
        Desired probability of avoiding stockout.

    Returns
    -------
    pandas.DataFrame
        Inventory planning table.
    """

    predictions = clean_forecast(
        predictions
    )

    historical_sales = np.asarray(
        historical_sales,
        dtype=float
    )

    current_stock = np.asarray(
        current_stock,
        dtype=float
    )

    if len(predictions) != len(historical_sales):
        raise ValueError(
            "predictions and historical_sales "
            "must contain the same number of rows."
        )

    if len(predictions) != len(current_stock):
        raise ValueError(
            "predictions and current_stock "
            "must contain the same number of rows."
        )

    current_stock = np.maximum(
        current_stock,
        0.0
    )

    demand_mean, demand_std = (
        calculate_demand_statistics(
            historical_sales
        )
    )

    safety_stock = calculate_safety_stock(
        demand_std=demand_std,
        lead_time=lead_time,
        service_level=service_level,
    )

    reorder_point = calculate_reorder_point(
        forecast_demand=predictions,
        safety_stock=safety_stock,
        lead_time=lead_time,
    )

    recommended_order = calculate_order_quantity(
        reorder_point=reorder_point,
        current_stock=current_stock,
    )

    recommendation = np.where(
        recommended_order > 0,
        "ORDER",
        "NO ORDER",
    )

    return pd.DataFrame(
        {
            "forecast_demand": predictions,
            "historical_mean_demand": demand_mean,
            "demand_std": demand_std,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "current_stock": current_stock,
            "recommended_order_qty": recommended_order,
            "recommendation": recommendation,
        }
    )


def calculate_single_product_plan(
    forecast_demand,
    historical_sales,
    current_stock,
    lead_time=1,
    service_level=0.95,
):
    """
    Calculate an inventory recommendation
    for one product/store.
    """

    forecast_demand = max(
        float(forecast_demand),
        0.0
    )

    historical_sales = np.asarray(
        historical_sales,
        dtype=float
    )

    current_stock = max(
        float(current_stock),
        0.0
    )

    demand_std = float(
        np.std(historical_sales)
    )

    safety_stock = float(
        calculate_safety_stock(
            demand_std,
            lead_time,
            service_level,
        )
    )

    reorder_point = float(
        calculate_reorder_point(
            forecast_demand,
            safety_stock,
            lead_time,
        )
    )

    order_quantity = int(
        calculate_order_quantity(
            reorder_point,
            current_stock,
        )
    )

    return {
        "forecast_demand": forecast_demand,
        "demand_std": demand_std,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "current_stock": current_stock,
        "recommended_order_qty": order_quantity,
        "recommendation": (
            "ORDER"
            if order_quantity > 0
            else "NO ORDER"
        ),
    }


if __name__ == "__main__":

    print("=" * 60)
    print("INVENTORY OPTIMIZATION MODULE")
    print("=" * 60)

    # Small demonstration
    historical_sales = np.array(
        [1, 3, 1, 1, 2, 1, 0, 0, 2, 0, 0],
        dtype=float,
    )

    forecast = 3.0
    current_stock = 1.0

    result = calculate_single_product_plan(
        forecast_demand=forecast,
        historical_sales=historical_sales,
        current_stock=current_stock,
    )

    print("\nExample product:")
    print(f"Forecast demand : {result['forecast_demand']:.2f} units")
    print(f"Demand std      : {result['demand_std']:.2f}")
    print(f"Safety stock    : {result['safety_stock']:.2f} units")
    print(f"Reorder point   : {result['reorder_point']:.2f} units")
    print(f"Current stock   : {result['current_stock']:.2f} units")
    print(
        f"Recommended order: "
        f"{result['recommended_order_qty']} units"
    )
    print(f"Recommendation  : {result['recommendation']}")

    print("\nInventory module ready.")