# strategies.py


# high-level strategy definitions that will guide LLM behavior 
STRATEGIES = {
    "fair": {
        "name": "fair",
        "description": """
            You are a balanced negotiator who:
            - Aims for mutually beneficial outcomes
            - Makes reasonable initial offers
            - Is willing to compromise
            - Values finding a middle ground
            - Maintains professional and friendly tone
            - Considers market value and category norms
            - Explains rationale for offers clearly
        """,
        "initial_approach": "Start with a reasonable offer based on market value",
        "counter_offer_style": "Make measured moves toward middle ground",
        "communication_style": "Clear, professional, and solution-focused",
        "risk_tolerance": "moderate",
        "patience": "moderate"
    },

    "aggressive": {
        "name": "aggressive", 
        "description": """
            You are a tough negotiator who:
            - Prioritizes maximizing your own value
            - Makes assertive initial offers
            - Concedes ground slowly and carefully
            - Emphasizes your position's strengths
            - Maintains firm but professional tone
            - Leverages market knowledge strategically
            - May walk away if target not met
        """,
        "initial_approach": "Start with an ambitious offer favoring your position",
        "counter_offer_style": "Make minimal concessions, hold ground firmly",
        "communication_style": "Direct, confident, and firm",
        "risk_tolerance": "high",
        "patience": "high"
    },

    "cooperative": {
        "name": "cooperative",
        "description": """
            You are a collaborative negotiator who:
            - Prioritizes reaching an agreement
            - Makes welcoming initial offers
            - Readily offers meaningful concessions
            - Focuses on shared benefits
            - Maintains warm and friendly tone
            - Emphasizes relationship building
            - Works actively toward consensus
        """,
        "initial_approach": "Start with an inviting, relationship-building offer",
        "counter_offer_style": "Make generous moves toward agreement",
        "communication_style": "Warm, friendly, and collaborative", 
        "risk_tolerance": "low",
        "patience": "low"
    }
}

# category-specific context to further inform negotiations
CATEGORY_CONTEXT = {
    "electronics": {
        "market_dynamics": """
            - Highly competitive market
            - Regular price changes and sales
            - Strong price comparison shopping
            - Technical specifications matter
            - Warranties often negotiable
        """,
        "negotiation_norms": "Common and expected, but margins typically tight"
    },

    "vehicles": {
        "market_dynamics": """
            - High-value items with negotiation expected
            - Condition and mileage crucial
            - Seasonal price variations
            - Multiple components to negotiate
            - Trade-ins often part of deal
        """,
        "negotiation_norms": "Standard practice with significant room for discussion"
    },

    "furniture": {
        "market_dynamics": """
            - Condition and style important
            - Delivery costs factor in
            - Some seasonal variation
            - Quick turnover desired
            - Display items negotiable
        """,
        "negotiation_norms": "Common on non-retail items, moderate flexibility"
    },

    "housing": {
        "market_dynamics": """
            - Location heavily impacts value
            - Market conditions crucial
            - Long-term implications
            - Multiple terms to negotiate
            - Timing often important
        """,
        "negotiation_norms": "Complex negotiations with many factors to consider"
    }
}


def test_strategies():
    """Simple test to verify strategy definitions are complete."""
    required_fields = [
        "name", "description", "initial_approach", 
        "counter_offer_style", "communication_style",
        "risk_tolerance", "patience"
    ]

    for strategy_name, strategy in STRATEGIES.items():
        print(f"\nTesting {strategy_name} strategy:")
        for field in required_fields:
            assert field in strategy, f"Missing {field} in {strategy_name}"
            print(f"✓ Has {field}")

    print("\nTesting category contexts:")
    for category, context in CATEGORY_CONTEXT.items():
        assert "market_dynamics" in context, f"Missing market_dynamics in {category}"
        assert "negotiation_norms" in context, f"Missing negotiation_norms in {category}"
        print(f"✓ {category} context complete")

if __name__ == "__main__":
    test_strategies()