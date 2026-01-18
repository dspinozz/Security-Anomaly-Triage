"""Rule-based baseline scoring for security events."""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class Rule:
    """A single scoring rule."""
    name: str
    description: str
    conditions: List[Dict]
    severity: str
    score_boost: float


@dataclass
class RuleConfig:
    """Configuration for baseline rules."""
    rules: List[Rule] = field(default_factory=list)
    severity_weights: Dict[str, float] = field(default_factory=dict)
    alert_threshold: float = 0.6
    critical_threshold: float = 0.85
    
    @classmethod
    def from_yaml(cls, path: Path) -> "RuleConfig":
        """Load rules from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        
        rules = [
            Rule(
                name=r["name"],
                description=r["description"],
                conditions=r["conditions"],
                severity=r["severity"],
                score_boost=r["score_boost"],
            )
            for r in config.get("rules", [])
        ]
        
        scoring_config = config.get("scoring", {})
        
        return cls(
            rules=rules,
            severity_weights=config.get("severity_weights", {}),
            alert_threshold=scoring_config.get("alert_threshold", 0.6),
            critical_threshold=scoring_config.get("critical_threshold", 0.85),
        )


class BaselineScorer:
    """Rule-based scorer using deterministic thresholds."""
    
    def __init__(self, config: Optional[RuleConfig] = None):
        """Initialize with rules configuration.
        
        Args:
            config: Rule configuration. If None, uses defaults.
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> RuleConfig:
        """Create default rule configuration."""
        return RuleConfig(
            rules=[
                Rule(
                    name="high_event_volume",
                    description="Unusually high event count",
                    conditions=[
                        {"field": "baseline_deviation_1min", "operator": ">", "value": 3.0}
                    ],
                    severity="medium",
                    score_boost=0.2,
                ),
                Rule(
                    name="port_scan",
                    description="Potential port scanning",
                    conditions=[
                        {"field": "unique_dst_ports_1min", "operator": ">", "value": 50},
                        {"field": "unique_dst_ips_1min", "operator": "<", "value": 5},
                    ],
                    severity="high",
                    score_boost=0.3,
                ),
                Rule(
                    name="high_port_entropy",
                    description="High port distribution entropy",
                    conditions=[
                        {"field": "port_entropy_1min", "operator": ">", "value": 5.0}
                    ],
                    severity="medium",
                    score_boost=0.15,
                ),
            ],
            severity_weights={"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2},
            alert_threshold=0.6,
            critical_threshold=0.85,
        )
    
    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score features using rules.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            DataFrame with added columns:
                - baseline_score: Rule-based score [0, 1]
                - triggered_rules: List of triggered rule names
                - severity: Max severity of triggered rules
        """
        results = []
        
        for idx, row in features.iterrows():
            triggered = []
            total_boost = 0.0
            max_severity = "none"
            
            for rule in self.config.rules:
                if self._evaluate_rule(rule, row):
                    triggered.append(rule.name)
                    total_boost += rule.score_boost
                    
                    # Track max severity
                    rule_weight = self.config.severity_weights.get(rule.severity, 0)
                    current_weight = self.config.severity_weights.get(max_severity, 0)
                    if rule_weight > current_weight:
                        max_severity = rule.severity
            
            results.append({
                "baseline_score": min(total_boost, 1.0),
                "triggered_rules": triggered,
                "severity": max_severity,
                "rule_count": len(triggered),
            })
        
        result_df = pd.DataFrame(results, index=features.index)
        return pd.concat([features, result_df], axis=1)
    
    def _evaluate_rule(self, rule: Rule, row: pd.Series) -> bool:
        """Evaluate if a rule triggers for a given row."""
        for condition in rule.conditions:
            field = condition["field"]
            operator = condition["operator"]
            threshold = condition["value"]
            
            if field not in row.index:
                return False
            
            value = row[field]
            
            if pd.isna(value):
                return False
            
            if operator == ">":
                if not (value > threshold):
                    return False
            elif operator == ">=":
                if not (value >= threshold):
                    return False
            elif operator == "<":
                if not (value < threshold):
                    return False
            elif operator == "<=":
                if not (value <= threshold):
                    return False
            elif operator == "==":
                if not (value == threshold):
                    return False
            elif operator == "!=":
                if not (value != threshold):
                    return False
        
        return True
    
    def get_rule_names(self) -> List[str]:
        """Get list of all rule names."""
        return [r.name for r in self.config.rules]
    
    def explain_triggers(
        self, 
        row: pd.Series,
    ) -> List[Dict]:
        """Explain which rules triggered and why.
        
        Args:
            row: Single row of features
            
        Returns:
            List of dicts with rule name, description, and condition values
        """
        explanations = []
        
        for rule in self.config.rules:
            if self._evaluate_rule(rule, row):
                condition_values = []
                for cond in rule.conditions:
                    field = cond["field"]
                    if field in row.index:
                        condition_values.append({
                            "field": field,
                            "operator": cond["operator"],
                            "threshold": cond["value"],
                            "actual_value": row[field],
                        })
                
                explanations.append({
                    "rule_name": rule.name,
                    "description": rule.description,
                    "severity": rule.severity,
                    "score_boost": rule.score_boost,
                    "conditions": condition_values,
                })
        
        return explanations
