import numpy as np

class MarketImpactSimulator:
    def __init__(self, temporary_impact, permanent_impact):
        self.temporary_impact = temporary_impact
        self.permanent_impact = permanent_impact
        
    def calculate_impact(self, current_price, order_size, total_volume):
        """
        Calculate price impact based on Almgren-Chriss model
        """
        # Normalize order size by market volume
        normalized_size = order_size / total_volume
        
        # Temporary impact (affects execution price)
        temp_impact = current_price * self.temporary_impact * normalized_size
        
        # Permanent impact (affects subsequent prices)
        perm_impact = current_price * self.permanent_impact * normalized_size
        
        execution_price = current_price + temp_impact
        new_market_price = current_price + perm_impact
        
        return execution_price, new_market_price