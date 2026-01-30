"""
Parameter Validator for simulation parameters.

This module provides validation functionality for simulation parameters
including type checking, range validation, and custom constraints.
"""

from typing import Any, Dict, List, Tuple, Optional, Callable


class ParameterValidator:
    """
    Validates simulation parameters against defined rules.
    
    Features:
    - Type validation with automatic conversion
    - Range validation (min/max)
    - Custom validation functions
    - Detailed error reporting
    
    Example:
        >>> rules = {
        ...     'Tx': {'min': 10e-9, 'max': 200e-9, 'type': float},
        ...     'Tz': {'min': 10e-9, 'max': 200e-9, 'type': float},
        ...     'alpha': {'min': 0.01, 'max': 1.0, 'type': float}
        ... }
        >>> validator = ParameterValidator(rules)
        >>> valid, errors = validator.validate({'Tx': 20e-9, 'Tz': 15e-9})
    """
    
    def __init__(self, rules: Dict[str, Dict[str, Any]]):
        """
        Initialize validator with parameter rules.
        
        Args:
            rules: Dictionary mapping parameter names to constraint dictionaries.
                   Each constraint dict can contain:
                   - 'type': Expected type (e.g., float, int, str)
                   - 'min': Minimum value (for numeric types)
                   - 'max': Maximum value (for numeric types)
                   - 'required': Boolean, whether parameter is required (default: True)
                   - 'validator': Custom validation function (param_value) -> bool
                   
        Example:
            rules = {
                'Tx': {
                    'type': float,
                    'min': 1e-9,
                    'max': 200e-9,
                    'required': True
                },
                'alpha': {
                    'type': float,
                    'min': 0.0,
                    'max': 1.0,
                    'validator': lambda x: x > 0  # Must be positive
                }
            }
        """
        self.rules = rules
    
    def validate(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates parameters against defined rules.
        
        Performs type conversion where possible and validates ranges.
        Modifies params dict in-place for successful type conversions.
        
        Args:
            params: Dictionary of parameter name -> value pairs to validate
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
            - is_valid: True if all validations pass
            - list_of_error_messages: List of validation error descriptions
            
        Example:
            >>> validator = ParameterValidator({'Tx': {'min': 10e-9, 'max': 200e-9, 'type': float}})
            >>> valid, errors = validator.validate({'Tx': 5e-9})
            >>> print(errors)
            ['Tx=5e-09 < min=1e-08']
        """
        errors = []
        
        for param_name, constraints in self.rules.items():
            # Check if parameter is required
            required = constraints.get('required', True)
            
            if param_name not in params:
                if required:
                    errors.append(f"Missing required parameter: {param_name}")
                continue
            
            value = params[param_name]
            
            # Type validation and conversion
            expected_type = constraints.get('type')
            if expected_type is not None:
                if not isinstance(value, expected_type):
                    # Try to convert
                    try:
                        converted_value = expected_type(value)
                        params[param_name] = converted_value  # Update in place
                        value = converted_value
                    except (ValueError, TypeError):
                        errors.append(
                            f"{param_name}: expected {expected_type.__name__}, "
                            f"got {type(value).__name__} (value={value})"
                        )
                        continue
            
            # Range validation (for numeric types)
            if isinstance(value, (int, float)):
                min_val = constraints.get('min')
                max_val = constraints.get('max')
                
                if min_val is not None and value < min_val:
                    errors.append(
                        f"{param_name}={value} < min={min_val}"
                    )
                
                if max_val is not None and value > max_val:
                    errors.append(
                        f"{param_name}={value} > max={max_val}"
                    )
            
            # Custom validator function
            custom_validator = constraints.get('validator')
            if custom_validator is not None:
                try:
                    if not custom_validator(value):
                        validator_name = getattr(custom_validator, '__name__', 'custom')
                        errors.append(
                            f"{param_name}={value} failed custom validation: {validator_name}"
                        )
                except Exception as e:
                    errors.append(
                        f"{param_name}: custom validator raised exception: {e}"
                    )
        
        return len(errors) == 0, errors
    
    def validate_single(self, param_name: str, value: Any) -> Tuple[bool, List[str]]:
        """
        Validates a single parameter.
        
        Args:
            param_name: Name of the parameter to validate
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
            
        Raises:
            KeyError: If param_name is not in rules
        """
        if param_name not in self.rules:
            raise KeyError(f"No validation rules defined for parameter: {param_name}")
        
        # Create a temporary dict with just this parameter
        temp_params = {param_name: value}
        temp_rules = {param_name: self.rules[param_name]}
        
        # Create temporary validator and validate
        temp_validator = ParameterValidator(temp_rules)
        return temp_validator.validate(temp_params)
    
    def get_required_params(self) -> List[str]:
        """
        Returns list of required parameter names.
        
        Returns:
            List of parameter names marked as required
        """
        return [
            name for name, constraints in self.rules.items()
            if constraints.get('required', True)
        ]
    
    def get_optional_params(self) -> List[str]:
        """
        Returns list of optional parameter names.
        
        Returns:
            List of parameter names marked as optional
        """
        return [
            name for name, constraints in self.rules.items()
            if not constraints.get('required', True)
        ]
    
    def add_rule(self, param_name: str, constraints: Dict[str, Any]) -> None:
        """
        Adds or updates a validation rule.
        
        Args:
            param_name: Name of the parameter
            constraints: Constraint dictionary (same format as __init__)
        """
        self.rules[param_name] = constraints
    
    def remove_rule(self, param_name: str) -> None:
        """
        Removes a validation rule.
        
        Args:
            param_name: Name of the parameter to remove
            
        Raises:
            KeyError: If param_name is not in rules
        """
        if param_name not in self.rules:
            raise KeyError(f"No rule defined for parameter: {param_name}")
        del self.rules[param_name]
    
    def __repr__(self) -> str:
        param_names = sorted(self.rules.keys())
        return f"ParameterValidator(params={param_names})"
