"""
Template Manager for MuMax3 simulation scripts.

This module provides functionality to manage and render MuMax3 template files
with parameterized placeholders.
"""

import os
import re
from typing import Set


class TemplateManager:
    """
    Manages MuMax3 template files with parameter substitution.
    
    Features:
    - Validates template file existence at initialization
    - Extracts required parameters from template
    - Renders template with parameter validation
    - Prevents missing parameter errors
    
    Example:
        >>> tm = TemplateManager("template.mx3")
        >>> print(tm.get_required_params())
        {'Tx', 'Tz'}
        >>> code = tm.render(Tx=20e-9, Tz=15e-9)
    """
    
    def __init__(self, template_path: str):
        """
        Initialize TemplateManager with a template file.
        
        Args:
            template_path: Path to the MuMax3 template file (.mx3)
            
        Raises:
            FileNotFoundError: If template file does not exist
        """
        self.template_path = template_path
        self._validate_template()
        self._required_params = self._extract_required_params()
    
    def _validate_template(self) -> None:
        """
        Validates that the template file exists.
        
        Raises:
            FileNotFoundError: If template file does not exist
        """
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(
                f"Template file not found: {self.template_path}\n"
                f"Please ensure the template file exists at the specified path."
            )
    
    def _extract_required_params(self) -> Set[str]:
        """
        Extracts required parameter names from template placeholders.
        
        Scans the template file for placeholders in the format {param_name}
        and returns a set of all unique parameter names.
        
        Returns:
            Set of parameter names found in the template
            
        Example:
            Template contains: "Tex := {Tx}" and "Tez := {Tz}"
            Returns: {'Tx', 'Tz'}
        """
        try:
            with open(self.template_path, 'r') as f:
                content = f.read()
            
            # Find all placeholders in format {parameter_name}
            matches = re.findall(r'\{(\w+)\}', content)
            return set(matches)
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract parameters from template: {e}"
            )
    
    def render(self, **params) -> str:
        """
        Renders the template by replacing placeholders with parameter values.
        
        Args:
            **params: Keyword arguments containing parameter values
            
        Returns:
            Rendered template content as string
            
        Raises:
            ValueError: If required parameters are missing
            IOError: If template file cannot be read
            
        Example:
            >>> tm = TemplateManager("template.mx3")
            >>> code = tm.render(Tx=20e-9, Tz=15e-9, alpha=0.1)
        """
        # Validate: check if all required parameters are provided
        missing = self._required_params - set(params.keys())
        if missing:
            raise ValueError(
                f"Missing required parameters: {sorted(missing)}\n"
                f"Required: {sorted(self._required_params)}\n"
                f"Provided: {sorted(params.keys())}"
            )
        
        # Read template
        try:
            with open(self.template_path, 'r') as f:
                content = f.read()
        except Exception as e:
            raise IOError(f"Failed to read template file: {e}")
        
        # Replace placeholders with values
        for key, value in params.items():
            placeholder = f'{{{key}}}'
            content = content.replace(placeholder, str(value))
        
        return content
    
    def get_required_params(self) -> Set[str]:
        """
        Returns a copy of the set of required parameter names.
        
        Returns:
            Set of parameter names extracted from template
        """
        return self._required_params.copy()
    
    def validate_params(self, **params) -> tuple[bool, list[str]]:
        """
        Validates parameters without rendering the template.
        
        Useful for pre-flight checks before batch processing.
        
        Args:
            **params: Keyword arguments to validate
            
        Returns:
            Tuple of (is_valid, list_of_missing_params)
            
        Example:
            >>> tm = TemplateManager("template.mx3")
            >>> valid, missing = tm.validate_params(Tx=20e-9)
            >>> if not valid:
            ...     print(f"Missing: {missing}")
        """
        missing = self._required_params - set(params.keys())
        return len(missing) == 0, sorted(missing)
    
    def __repr__(self) -> str:
        return (
            f"TemplateManager(template_path='{self.template_path}', "
            f"required_params={sorted(self._required_params)})"
        )
