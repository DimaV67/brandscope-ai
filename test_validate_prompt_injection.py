# src/stage2/prompt_template_injector.py
"""
Stage 2 Prompt Template Variable Injection System
UPDATED: Now includes required JSON output specifications from Pilot_mode_HM_V3-styled.txt
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from string import Template

from src.utils.logging import get_logger
from src.utils.exceptions import StageExecutionError
from src.stage2.stage1_data_loader import Stage1Data


logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Structured prompt template with metadata"""
    name: str
    template: str
    required_variables: List[str]
    description: str
    prompt_type: str  # 'natural_ai' or 'controlled_ai'


@dataclass
class InjectedPrompt:
    """Prompt with variables injected, ready for execution"""
    prompt_id: str
    query_id: str
    prompt_type: str
    styled_query: str
    customer_context: str
    injected_prompt: str
    platforms: List[str]
    execution_priority: int
    archetype: str
    expected_response_file: str
    metadata: Dict[str, Any]


class PromptTemplateManager:
    """Manages prompt templates with required JSON output specifications"""
    
    def __init__(self):
        self.templates = self._load_document_templates()
    
    def _load_document_templates(self) -> Dict[str, PromptTemplate]:
        """Load exact prompt templates with required JSON outputs from specification."""
        
        return {
            'natural_ai': PromptTemplate(
                name="Natural AI Testing",
                template="""You are a helpful shopping assistant helping customers make product decisions. Provide useful, actionable advice.

CRITICAL OUTPUT STRUCTURE: Your entire output must be a single, continuous response. It will contain TWO required parts: 1) The user-facing answer, followed by 2) The complete JSON analysis of your answer. Do not omit the JSON block.

Be helpful and confident in your recommendations. Use your knowledge to provide specific guidance. Address the customer's constraints and priorities. Give practical shopping advice.

Customer context: ${customer_context}

Customer query: ${styled_query}

===== RESPONSE START =====
[Your natural, helpful response to the customer query goes here]
===== RESPONSE END =====

MANDATORY ANALYSIS: After the ===== RESPONSE END ===== marker, you must complete the following JSON analysis of your own response. This is a required and non-optional part of your output.

===== ANALYSIS START =====
Now analyze your response above and complete this JSON structure with actual data from your response:

{
  "response_metadata": {
    "word_count": [Count words in your response section above],
    "response_confidence_level": "HIGH|MEDIUM|LOW",
    "specific_claims_made": [Count specific claims you made in your response],
    "response_timestamp": "[Current timestamp in ISO format]",
    "dataset_type": "NATURAL_AI"
  },
  "recommendation_patterns": {
    "first_mentioned_brand": "[Name of first brand mentioned in your response, or null]",
    "mention_order": [List all brands mentioned in order, or []],
    "recommendation_strength_language": {
      "strong_indicators": [Extract phrases like "highly recommend", "best choice" from your response],
      "moderate_indicators": [Extract phrases like "good option", "consider" from your response],
      "weak_indicators": [Extract phrases like "might work", "depends on" from your response]
    },
    "conditional_recommendations": {
      "if_budget_focused": "[Brand you recommended for budget customers, or null]",
      "if_premium_seeking": "[Brand you recommended for premium customers, or null]",
      "if_convenience_priority": "[Brand you recommended for convenience, or null]",
      "if_health_focused": "[Brand you recommended for health, or null]"
    },
    "purchase_readiness_signals": [Extract buying intent phrases from your response, or []]
  },
  "brand_positioning_intelligence": {
    "brand_attribute_associations": {
      "wonderful_pistachios": [List attributes you linked to Wonderful, or []],
      "santa_barbara_pistachio": [List attributes you linked to this brand, or []],
      "365_whole_foods": [List attributes you linked to this brand, or []],
      "eden_foods": [List attributes you linked to this brand, or []]
    },
    "positive_qualifier_distribution": {
      "wonderful": [List positive words you used for Wonderful, or []],
      "competitors": [List positive words you used for competitors, or []]
    },
    "negative_qualifier_patterns": {
      "wonderful": [List limitations you mentioned for Wonderful, or []],
      "competitors": [List limitations you mentioned for competitors, or []]
    },
    "information_depth_by_brand": {
      "wonderful": {"facts_provided": [Count facts about Wonderful], "detail_level": "HIGH|MEDIUM|LOW"},
      "competitors": {"facts_provided": [Count facts about competitors], "detail_level": "HIGH|MEDIUM|LOW"}
    }
  },
  "hallucination_risk_indicators": {
    "specific_prices_claimed": [List price claims you made without verification, or []],
    "availability_assertions": [List availability claims you made, or []],
    "unverifiable_nutritional_claims": [List nutrition data you stated, or []],
    "competitive_claims": [List comparative statements you made, or []],
    "confidence_without_sources": [List confident claims without attribution, or []]
  },
  "natural_ai_behavior_patterns": {
    "tendency_to_hallucinate": "HIGH|MEDIUM|LOW",
    "confidence_vs_accuracy_balance": "overconfident|balanced|cautious",
    "brand_favoritism_indicators": [List bias patterns in your response, or []],
    "customer_satisfaction_optimization": "prioritizes helpful over accurate"
  },
  "commercial_intelligence": {
    "upsell_opportunities_presented": [List upselling you mentioned, or []],
    "purchase_size_guidance": "[Quote size/quantity guidance, or null]",
    "loyalty_building_language": [List loyalty phrases you used, or []],
    "cross_sell_indicators": [List related products suggested, or []]
  },
  "persona_alignment": {
    "health_focus_addressed": [true or false],
    "urgency_accommodated": [true or false],
    "budget_considered": [true or false],
    "location_context_used": [true or false],
    "missed_persona_opportunities": [List missed opportunities from customer profile, or []]
  }
}
===== ANALYSIS END =====""",
                required_variables=['customer_context', 'styled_query'],
                description="Natural AI behavior testing without accuracy controls, with comprehensive JSON analysis",
                prompt_type="natural_ai"
            ),
            
            'controlled_ai': PromptTemplate(
                name="Controlled AI Testing",
                template="""You are a knowledgeable shopping assistant helping customers make informed product decisions. Provide helpful, accurate responses with rigorous source attribution.

CRITICAL OUTPUT STRUCTURE: Your entire output must be a single, continuous response. It will contain TWO required parts: 1) The user-facing answer, followed by 2) The complete JSON analysis of your answer. Do not omit the JSON block.

SEARCH MANDATE: Use web search for ALL verifiable claims about:
- Current pricing and availability
- Product specifications and certifications
- Store-specific information
- Nutritional data and comparisons
- Brand feature comparisons

HALLUCINATION PREVENTION:
- NO specific prices without current search verification
- NO availability claims without real-time confirmation
- NO nutritional data without official source verification
- NO store policies without current validation
- If search returns no results: State "Unable to verify [specific claim]"

CONFIDENCE REQUIREMENTS:
- HIGH CONFIDENCE ONLY (>90%): Specific prices, availability, certifications
- MEDIUM CONFIDENCE WITH CAVEATS (70-90%): General features, typical patterns
- LOW CONFIDENCE WITH UNCERTAINTY (<70%): Use "generally", "typically", "may vary"

SOURCE ATTRIBUTION REQUIRED:
- Pricing: Store websites, retailer listings
- Nutrition: Brand websites, USDA database
- Availability: Store locators, product listings
- Certifications: Official certification databases

RESPONSE FORMAT:
- VERIFIED CLAIMS: "According to [source] at [URL], [specific claim]"
- GENERAL GUIDANCE: "Based on general knowledge, [guidance] (specific details may vary)"
- UNVERIFIABLE: "I cannot verify current [specific claim] - recommend checking [relevant source]"

PROHIBITED: Never state specific prices, availability, or nutritional values without current verification.

Customer context: ${customer_context}

Customer query: ${styled_query}

===== RESPONSE START =====
[Your helpful, verified response to the customer query goes here]
===== RESPONSE END =====

MANDATORY ANALYSIS: After the ===== RESPONSE END ===== marker, you must complete the following JSON analysis of your own response. This is a required and non-optional part of your output.

===== ANALYSIS START =====
Now analyze your response above and complete this JSON structure with actual data from your response:

{
  "response_metadata": {
    "word_count": [Count words in your response section above],
    "search_queries_used": [List the actual search queries you used, or [] if none],
    "sources_consulted": [List actual sources you referenced, or [] if none],
    "response_timestamp": "[Current timestamp in ISO format]",
    "dataset_type": "CONTROLLED_AI"
  },
  "recommendation_patterns": {
    "first_mentioned_brand": "[Name of first brand mentioned in your response, or null]",
    "mention_order": [List all brands mentioned in order, or []],
    "recommendation_strength_language": {
      "strong_indicators": [Extract phrases like "highly recommend", "best choice" from your response],
      "moderate_indicators": [Extract phrases like "good option", "consider" from your response],
      "weak_indicators": [Extract phrases like "might work", "depends on" from your response]
    },
    "conditional_recommendations": {
      "if_budget_focused": "[Brand you recommended for budget customers, or null]",
      "if_premium_seeking": "[Brand you recommended for premium customers, or null]",
      "if_convenience_priority": "[Brand you recommended for convenience, or null]",
      "if_health_focused": "[Brand you recommended for health, or null]"
    },
    "purchase_readiness_signals": [Extract buying intent phrases from your response, or []]
  },
  "brand_positioning_intelligence": {
    "brand_attribute_associations": {
      "wonderful_pistachios": [List attributes you linked to Wonderful, or []],
      "santa_barbara_pistachio": [List attributes you linked to this brand, or []],
      "365_whole_foods": [List attributes you linked to this brand, or []],
      "eden_foods": [List attributes you linked to this brand, or []]
    },
    "positive_qualifier_distribution": {
      "wonderful": [List positive words you used for Wonderful, or []],
      "competitors": [List positive words you used for competitors, or []]
    },
    "negative_qualifier_patterns": {
      "wonderful": [List limitations you mentioned for Wonderful, or []],
      "competitors": [List limitations you mentioned for competitors, or []]
    },
    "information_depth_by_brand": {
      "wonderful": {"facts_provided": [Count facts about Wonderful], "detail_level": "HIGH|MEDIUM|LOW"},
      "competitors": {"facts_provided": [Count facts about competitors], "detail_level": "HIGH|MEDIUM|LOW"}
    }
  },
  "accuracy_and_verification": {
    "claims_with_sources": [Count how many claims you backed with sources],
    "claims_without_sources": [Count claims you made without sources],
    "uncertainty_markers_used": [Count times you used "generally", "typically", etc.],
    "search_verification_success_rate": [Decimal between 0.0-1.0],
    "information_gaps_acknowledged": [List areas where you admitted uncertainty, or []],
    "overall_accuracy_confidence": "HIGH|MEDIUM|LOW"
  },
  "controlled_ai_behavior_patterns": {
    "search_compliance": "HIGH|MEDIUM|LOW",
    "source_attribution_quality": "comprehensive|partial|minimal|none",
    "uncertainty_acknowledgment": "appropriate|insufficient|excessive",
    "verification_thoroughness": "rigorous|moderate|minimal"
  },
  "claims_analysis": [
    {
      "claim": "[Quote a specific claim from your response]",
      "claim_type": "PRICE|NUTRITION|AVAILABILITY|FEATURE|COMPARISON|BENEFIT",
      "source_type": "SEARCH_RESULT|WEB_FETCH|GENERAL_KNOWLEDGE",
      "source_url": "[URL if you cited one, or null]",
      "confidence": "HIGH|MEDIUM|LOW",
      "persona_relevance": "[How this serves the customer]",
      "brand_association": "[Which brand this supports]",
      "verification_status": "VERIFIED|UNVERIFIED|UNCERTAIN"
    }
  ],
  "commercial_intelligence": {
    "upsell_opportunities_presented": [List upselling you mentioned, or []],
    "purchase_size_guidance": "[Quote size/quantity guidance, or null]",
    "loyalty_building_language": [List loyalty phrases you used, or []],
    "cross_sell_indicators": [List related products suggested, or []],
    "premium_tier_positioning": {
      "organic_upgrade_mentioned": [true or false],
      "premium_justification": "[Your reasoning for premium options, or null]"
    }
  },
  "competitive_landscape_analysis": {
    "market_share_implications": {
      "wonderful_share_of_voice": [Estimate 0.0-1.0 based on how much you talked about Wonderful],
      "competitor_collective_voice": [Estimate 0.0-1.0 for all competitors combined],
      "recommendation_hierarchy": [Order brands by recommendation strength, or []]
    },
    "differentiation_messaging": {
      "wonderful_unique_positioning": [List unique attributes you highlighted for Wonderful, or []],
      "competitor_advantages_acknowledged": [List competitor advantages you mentioned, or []],
      "parity_claims": [List where you said brands are equivalent, or []]
    }
  },
  "persona_alignment": {
    "health_focus_addressed": [true or false],
    "urgency_accommodated": [true or false],
    "budget_considered": [true or false],
    "location_context_used": [true or false],
    "missed_persona_opportunities": [List missed opportunities, or []]
  },
  "strategic_insights": {
    "immediate_optimization_opportunities": [List optimization opportunities, or []],
    "competitive_vulnerabilities_exposed": [List competitive weaknesses revealed, or []],
    "customer_journey_momentum": "ADVANCING|STALLED|REDIRECTED",
    "next_likely_customer_questions": [List likely follow-up questions, or []],
    "conversion_probability_indicators": "HIGH|MEDIUM|LOW"
  }
}
===== ANALYSIS END =====""",
                required_variables=['customer_context', 'styled_query'],
                description="Controlled AI testing with enterprise-grade accuracy requirements and comprehensive JSON analysis",
                prompt_type="controlled_ai"
            )
        }
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific prompt template."""
        if template_name not in self.templates:
            raise StageExecutionError(
                f"Template '{template_name}' not found. Available: {list(self.templates.keys())}",
                stage="stage2",
                operation="get_template"
            )
        return self.templates[template_name]
    
    def validate_template_variables(self, template_name: str, variables: Dict[str, str]) -> List[str]:
        """Validate that all required variables are provided."""
        template = self.get_template(template_name)
        missing_vars = []
        
        for required_var in template.required_variables:
            if required_var not in variables or not variables[required_var]:
                missing_vars.append(required_var)
        
        return missing_vars


class PromptVariableInjector:
    """Injects variables into prompt templates to create executable prompts"""
    
    def __init__(self):
        self.template_manager = PromptTemplateManager()
    
    def inject_variables(
        self, 
        stage1_data: Stage1Data,
        priority_queries: List[Dict[str, Any]],
        customer_context: str,
        platforms: List[str] = None
    ) -> Tuple[List[InjectedPrompt], List[InjectedPrompt]]:
        """
        Inject variables into templates to create executable prompts with JSON outputs.
        
        Args:
            stage1_data: Validated Stage 1 data
            priority_queries: Selected priority queries for execution
            customer_context: Customer context string
            platforms: Target AI platforms
            
        Returns:
            Tuple of (natural_prompts, controlled_prompts) with JSON analysis requirements
        """
        if platforms is None:
            platforms = ['claude', 'chatgpt', 'gemini', 'grok']
        
        logger.info("Starting prompt variable injection with JSON outputs",
                   metadata={
                       "queries_count": len(priority_queries),
                       "platforms_count": len(platforms),
                       "customer_context_length": len(customer_context),
                       "json_analysis_required": True
                   })
        
        natural_prompts = []
        controlled_prompts = []
        
        for i, query in enumerate(priority_queries, 1):
            # Generate natural AI prompt with JSON analysis requirements
            natural_prompt = self._inject_single_prompt(
                query=query,
                customer_context=customer_context,
                template_type="natural_ai",
                platforms=platforms,
                execution_priority=i
            )
            natural_prompts.append(natural_prompt)
            
            # Generate controlled AI prompt with JSON analysis requirements
            controlled_prompt = self._inject_single_prompt(
                query=query,
                customer_context=customer_context,
                template_type="controlled_ai",
                platforms=platforms,
                execution_priority=i
            )
            controlled_prompts.append(controlled_prompt)
        
        logger.info("Prompt variable injection completed with JSON requirements",
                   metadata={
                       "natural_prompts": len(natural_prompts),
                       "controlled_prompts": len(controlled_prompts),
                       "total_prompts": len(natural_prompts) + len(controlled_prompts),
                       "json_analysis_included": True
                   })
        
        return natural_prompts, controlled_prompts
    
    def _inject_single_prompt(
        self,
        query: Dict[str, Any],
        customer_context: str,
        template_type: str,
        platforms: List[str],
        execution_priority: int
    ) -> InjectedPrompt:
        """Inject variables into a single prompt template with JSON analysis."""
        
        template = self.template_manager.get_template(template_type)
        
        # Prepare variables for injection
        variables = {
            'customer_context': customer_context,
            'styled_query': query.get('styled_query', '')
        }
        
        # Validate required variables
        missing_vars = self.template_manager.validate_template_variables(template_type, variables)
        if missing_vars:
            raise StageExecutionError(
                f"Missing required variables for {template_type}: {missing_vars}",
                stage="stage2",
                operation="inject_variables"
            )
        
        # Inject variables using Template
        template_obj = Template(template.template)
        try:
            injected_prompt = template_obj.substitute(variables)
        except KeyError as e:
            raise StageExecutionError(
                f"Template variable substitution failed: {e}",
                stage="stage2",
                operation="inject_variables"
            ) from e
        
        # Generate prompt ID and metadata
        query_id = query.get('query_id', f'Q{execution_priority:03d}')
        prompt_id = f"{template_type.upper()}_{query_id}"
        
        return InjectedPrompt(
            prompt_id=prompt_id,
            query_id=query_id,
            prompt_type=template_type,
            styled_query=query.get('styled_query', ''),
            customer_context=customer_context,
            injected_prompt=injected_prompt,
            platforms=platforms.copy(),
            execution_priority=execution_priority,
            archetype=query.get('archetype', 'Unknown'),
            expected_response_file=self._generate_response_filename(
                query_id, template_type, platforms
            ),
            metadata={
                'template_name': template.name,
                'query_authenticity': query.get('authenticity_score', 0.0),
                'query_category': query.get('category', 'unknown'),
                'injection_timestamp': datetime.now().isoformat(),
                'archetype_id': query.get('archetype_id', 'UNKNOWN'),
                'json_analysis_required': True,
                'analysis_sections': self._get_analysis_sections(template_type)
            }
        )
    
    def _generate_response_filename(
        self, 
        query_id: str, 
        template_type: str, 
        platforms: List[str]
    ) -> List[str]:
        """Generate expected response filenames for manual execution."""
        
        # Extract query number from query_id (e.g., "Q004" -> "4")
        import re
        query_match = re.search(r'(\d+)', query_id)
        query_num = query_match.group(1) if query_match else "01"
        
        # Convert template type to filename format
        prompt_type = "natural" if template_type == "natural_ai" else "controlled"
        
        # Generate filenames for each platform
        filenames = []
        for platform in platforms:
            filename = f"{platform}_query{query_num}_{prompt_type}.json"
            filenames.append(filename)
        
        return filenames
    
    def _get_analysis_sections(self, template_type: str) -> List[str]:
        """Get required analysis sections for the template type."""
        
        natural_sections = [
            "response_metadata",
            "recommendation_patterns", 
            "brand_positioning_intelligence",
            "hallucination_risk_indicators",
            "natural_ai_behavior_patterns",
            "commercial_intelligence",
            "persona_alignment"
        ]
        
        controlled_sections = [
            "response_metadata",
            "recommendation_patterns",
            "brand_positioning_intelligence", 
            "accuracy_and_verification",
            "controlled_ai_behavior_patterns",
            "claims_analysis",
            "commercial_intelligence",
            "competitive_landscape_analysis",
            "persona_alignment",
            "strategic_insights"
        ]
        
        return natural_sections if template_type == "natural_ai" else controlled_sections


class ExecutionPackageGenerator:
    """Generates complete execution packages for manual Stage 2 testing"""
    
    def __init__(self):
        self.injector = PromptVariableInjector()
    
    def generate_execution_package(
        self,
        stage1_data: Stage1Data,
        priority_queries: List[Dict[str, Any]],
        customer_context: str,
        project_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate complete execution package with injected prompts and JSON requirements.
        
        Args:
            stage1_data: Validated Stage 1 data
            priority_queries: Selected priority queries
            customer_context: Customer context string
            project_metadata: Project metadata from Stage 1
            
        Returns:
            Complete execution package ready for manual testing with JSON analysis
        """
        logger.info("Generating execution package with JSON analysis requirements",
                   metadata={"queries_count": len(priority_queries)})
        
        # Inject variables into prompt templates
        natural_prompts, controlled_prompts = self.injector.inject_variables(
            stage1_data=stage1_data,
            priority_queries=priority_queries,
            customer_context=customer_context
        )
        
        # Create execution matrix
        execution_matrix = self._create_execution_matrix(natural_prompts, controlled_prompts)
        
        # Generate file structure expectations
        file_structure = self._generate_file_structure(natural_prompts, controlled_prompts)
        
        # Create quality control checklist with JSON requirements
        quality_checklist = self._generate_quality_checklist(natural_prompts, controlled_prompts)
        
        # Create JSON analysis guide
        json_analysis_guide = self._generate_json_analysis_guide()
        
        execution_package = {
            'metadata': {
                **project_metadata,
                'stage2_generation_timestamp': datetime.now().isoformat(),
                'prompts_generated': len(natural_prompts) + len(controlled_prompts),
                'json_analysis_required': True,
                'specification_compliance': 'Pilot_mode_HM_V3-styled.txt',
                'execution_ready': True
            },
            'customer_context': customer_context,
            'execution_matrix': execution_matrix,
            'natural_prompts': [self._prompt_to_dict(p) for p in natural_prompts],
            'controlled_prompts': [self._prompt_to_dict(p) for p in controlled_prompts],
            'file_structure': file_structure,
            'quality_control': quality_checklist,
            'json_analysis_guide': json_analysis_guide,
            'platforms': natural_prompts[0].platforms if natural_prompts else [],
            'execution_summary': {
                'total_queries': len(priority_queries),
                'total_prompts': len(natural_prompts) + len(controlled_prompts),
                'expected_response_files': len(priority_queries) * 4 * 2,  # queries × platforms × prompt_types
                'json_analysis_sections': self._count_analysis_sections(),
                'estimated_execution_time_minutes': len(priority_queries) * 4 * 8,  # 8 minutes per platform per query (including JSON analysis)
                'manual_execution_required': True,
                'specification_compliance': True
            }
        }
        
        logger.info("Execution package generated successfully with JSON requirements",
                   metadata={
                       "total_prompts": execution_package['metadata']['prompts_generated'],
                       "expected_files": execution_package['execution_summary']['expected_response_files'],
                       "json_analysis_included": True
                   })
        
        return execution_package
    
    def _create_execution_matrix(
        self, 
        natural_prompts: List[InjectedPrompt], 
        controlled_prompts: List[InjectedPrompt]
    ) -> List[Dict[str, Any]]:
        """Create execution matrix showing all prompt/platform combinations with JSON requirements."""
        
        execution_matrix = []
        
        for natural, controlled in zip(natural_prompts, controlled_prompts):
            matrix_entry = {
                'query_id': natural.query_id,
                'styled_query': natural.styled_query,
                'archetype': natural.archetype,
                'execution_priority': natural.execution_priority,
                'natural_prompt_id': natural.prompt_id,
                'controlled_prompt_id': controlled.prompt_id,
                'platforms': natural.platforms,
                'expected_files': {
                    'natural': natural.expected_response_file,
                    'controlled': controlled.expected_response_file
                },
                'json_analysis_required': True,
                'natural_analysis_sections': natural.metadata['analysis_sections'],
                'controlled_analysis_sections': controlled.metadata['analysis_sections'],
                'estimated_time_minutes': len(natural.platforms) * 8  # 8 minutes per platform including analysis
            }
            execution_matrix.append(matrix_entry)
        
        return execution_matrix
    
    def _generate_file_structure(
        self, 
        natural_prompts: List[InjectedPrompt], 
        controlled_prompts: List[InjectedPrompt]
    ) -> Dict[str, Any]:
        """Generate expected file structure for manual execution with JSON analysis."""
        
        all_natural_files = []
        all_controlled_files = []
        
        for prompt in natural_prompts:
            all_natural_files.extend(prompt.expected_response_file)
        
        for prompt in controlled_prompts:
            all_controlled_files.extend(prompt.expected_response_file)
        
        return {
            'stage2_execution': {
                'description': 'Manual execution workspace with JSON analysis requirements',
                'subdirectories': {
                    'natural_dataset': {
                        'description': 'Natural AI responses with comprehensive JSON analysis',
                        'expected_files': len(all_natural_files),
                        'file_list': all_natural_files,
                        'json_sections_required': 7,  # Natural AI has 7 analysis sections
                        'analysis_format': 'Pilot_mode_HM_V3-styled.txt specification'
                    },
                    'controlled_dataset': {
                        'description': 'Controlled AI responses with comprehensive JSON analysis',
                        'expected_files': len(all_controlled_files),
                        'file_list': all_controlled_files,
                        'json_sections_required': 10,  # Controlled AI has 10 analysis sections
                        'analysis_format': 'Pilot_mode_HM_V3-styled.txt specification'
                    }
                }
            },
            'naming_convention': '{platform}_query{number}_{type}.json',
            'total_expected_files': len(all_natural_files) + len(all_controlled_files),
            'json_analysis_compliance': True
        }
    
    def _generate_quality_checklist(
        self, 
        natural_prompts: List[InjectedPrompt], 
        controlled_prompts: List[InjectedPrompt]
    ) -> Dict[str, Any]:
        """Generate quality control checklist for manual execution with JSON requirements."""
        
        platforms = natural_prompts[0].platforms if natural_prompts else []
        
        return {
            'pre_execution_checks': [
                'All prompt templates properly injected with variables',
                'Customer context accurately reflects Stage 1 data',
                'Query priorities correctly assigned',
                'File naming convention understood',
                'JSON analysis requirements understood from specification',
                'Pilot_mode_HM_V3-styled.txt specification accessible'
            ],
            'during_execution_checks': [
                f'Test each query on all {len(platforms)} platforms',
                'Use natural prompt for natural dataset',
                'Use controlled prompt for controlled dataset',
                'Save responses with exact filename format',
                'Complete ALL required JSON analysis sections',
                'Follow ===== RESPONSE START/END ===== format exactly',
                'Complete ===== ANALYSIS START/END ===== sections with valid JSON',
                'Capture complete AI responses including any search results'
            ],
            'post_execution_checks': [
                f'Total files created: {len(natural_prompts) * len(platforms) * 2}',
                'All files contain both RESPONSE and ANALYSIS sections',
                'All JSON analysis sections are valid and complete',
                'Natural AI responses include 7 required analysis sections',
                'Controlled AI responses include 10 required analysis sections',
                'No missing responses for any platform/query combination',
                'Response metadata properly recorded',
                'Ready for Stage 3 comparative analysis processing'
            ],
            'json_analysis_validation': {
                'natural_ai_required_sections': [
                    'response_metadata',
                    'recommendation_patterns',
                    'brand_positioning_intelligence',
                    'hallucination_risk_indicators',
                    'natural_ai_behavior_patterns',
                    'commercial_intelligence',
                    'persona_alignment'
                ],
                'controlled_ai_required_sections': [
                    'response_metadata',
                    'recommendation_patterns',
                    'brand_positioning_intelligence',
                    'accuracy_and_verification',
                    'controlled_ai_behavior_patterns',
                    'claims_analysis',
                    'commercial_intelligence',
                    'competitive_landscape_analysis',
                    'persona_alignment',
                    'strategic_insights'
                ],
                'validation_requirements': {
                    'total_files_expected': len(natural_prompts) * len(platforms) * 2,
                    'platforms_tested': platforms,
                    'queries_tested': len(natural_prompts),
                    'prompt_types_tested': ['natural_ai', 'controlled_ai'],
                    'json_format_compliance': 'required',
                    'specification_adherence': 'Pilot_mode_HM_V3-styled.txt'
                }
            }
        }
    
    def _generate_json_analysis_guide(self) -> Dict[str, Any]:
        """Generate guide for completing JSON analysis sections."""
        
        return {
            'overview': 'Each response must include comprehensive JSON analysis as specified in Pilot_mode_HM_V3-styled.txt',
            'format_requirements': {
                'response_section': 'Natural AI response between ===== RESPONSE START/END =====',
                'analysis_section': 'Complete JSON analysis between ===== ANALYSIS START/END =====',
                'json_validation': 'All JSON must be valid and parseable',
                'completeness': 'All required sections must be filled with actual analysis data'
            },
            'natural_ai_analysis_sections': {
                'response_metadata': 'Word count, confidence level, claims count, timestamp, dataset type',
                'recommendation_patterns': 'Brand mentions, recommendation strength, conditional recommendations',
                'brand_positioning_intelligence': 'Brand associations, qualifiers, information depth',
                'hallucination_risk_indicators': 'Unverified claims, confidence without sources',
                'natural_ai_behavior_patterns': 'Hallucination tendency, confidence vs accuracy balance',
                'commercial_intelligence': 'Upsell opportunities, loyalty language, cross-sell indicators',
                'persona_alignment': 'How well response addresses customer profile'
            },
            'controlled_ai_analysis_sections': {
                'response_metadata': 'Word count, search queries used, sources consulted, timestamp',
                'recommendation_patterns': 'Same as natural AI',
                'brand_positioning_intelligence': 'Same as natural AI',
                'accuracy_and_verification': 'Claims with/without sources, uncertainty markers',
                'controlled_ai_behavior_patterns': 'Search compliance, source attribution quality',
                'claims_analysis': 'Individual claim verification and source analysis',
                'commercial_intelligence': 'Same as natural AI plus premium positioning',
                'competitive_landscape_analysis': 'Market share implications, differentiation messaging',
                'persona_alignment': 'Same as natural AI',
                'strategic_insights': 'Optimization opportunities, competitive vulnerabilities'
            },
            'data_quality_requirements': {
                'no_placeholders': 'Replace all placeholder values with actual analysis',
                'quantitative_data': 'Provide actual counts, percentages, and scores',
                'specific_examples': 'Include exact quotes and specific brand mentions',
                'boolean_values': 'Use true/false not true/false strings',
                'array_completeness': 'Fill arrays with actual content, not placeholder text'
            }
        }
    
    def _count_analysis_sections(self) -> Dict[str, int]:
        """Count analysis sections for each prompt type."""
        return {
            'natural_ai_sections': 7,
            'controlled_ai_sections': 10,
            'total_sections_per_query': 17
        }
    
    def _prompt_to_dict(self, prompt: InjectedPrompt) -> Dict[str, Any]:
        """Convert InjectedPrompt to dictionary for JSON serialization."""
        return {
            'prompt_id': prompt.prompt_id,
            'query_id': prompt.query_id,
            'prompt_type': prompt.prompt_type,
            'styled_query': prompt.styled_query,
            'customer_context': prompt.customer_context,
            'injected_prompt': prompt.injected_prompt,
            'platforms': prompt.platforms,
            'execution_priority': prompt.execution_priority,
            'archetype': prompt.archetype,
            'expected_response_files': prompt.expected_response_file,
            'metadata': prompt.metadata
        }


class ExecutionGuideGenerator:
    """Generates human-readable execution guides for manual testing with JSON requirements"""
    
    def __init__(self):
        self.package_generator = ExecutionPackageGenerator()
    
    def generate_execution_guide(
        self,
        execution_package: Dict[str, Any],
        project_metadata: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive human-readable execution guide with JSON analysis requirements.
        
        Args:
            execution_package: Complete execution package
            project_metadata: Project metadata
            
        Returns:
            Markdown-formatted execution guide with JSON analysis instructions
        """
        metadata = execution_package['metadata']
        natural_prompts = execution_package['natural_prompts']
        controlled_prompts = execution_package['controlled_prompts']
        execution_matrix = execution_package['execution_matrix']
        file_structure = execution_package['file_structure']
        json_guide = execution_package['json_analysis_guide']
        
        guide_content = f"""# Stage 2 Manual Execution Guide with JSON Analysis
## {metadata.get('brand', 'Brand')} {metadata.get('category', 'Category')} Analysis

**Generated**: {metadata['stage2_generation_timestamp']}  
**Project ID**: {metadata.get('project_id', 'Unknown')}  
**Correlation ID**: {metadata.get('correlation_id', 'Unknown')}  
**Specification**: {metadata.get('specification_compliance', 'Pilot_mode_HM_V3-styled.txt')}

---

## 📋 Execution Overview

### Customer Context
{execution_package['customer_context']}

### Execution Summary
- **Total Queries**: {len(natural_prompts)}
- **Total Prompts**: {len(natural_prompts) + len(controlled_prompts)} ({len(natural_prompts)} natural + {len(controlled_prompts)} controlled)
- **Platforms**: {', '.join(execution_package.get('platforms', []))}
- **Expected Response Files**: {execution_package['execution_summary']['expected_response_files']}
- **JSON Analysis Required**: ✅ Yes (per Pilot_mode_HM_V3-styled.txt)
- **Estimated Time**: {execution_package['execution_summary']['estimated_execution_time_minutes']} minutes (includes JSON analysis)

---

## 🎯 Query Execution Matrix

"""
        
        # Add execution matrix table with JSON requirements
        for i, matrix_entry in enumerate(execution_matrix, 1):
            guide_content += f"""### Query {i}: {matrix_entry['query_id']}
**Styled Query**: "{matrix_entry['styled_query']}"  
**Archetype**: {matrix_entry['archetype']}  
**Priority**: {matrix_entry['execution_priority']}  

**Natural Prompt ID**: `{matrix_entry['natural_prompt_id']}`  
**Controlled Prompt ID**: `{matrix_entry['controlled_prompt_id']}`  
**JSON Analysis Sections**: {len(matrix_entry['natural_analysis_sections'])} (Natural) + {len(matrix_entry['controlled_analysis_sections'])} (Controlled)

**Expected Files**:
"""
            # Add expected files for this query
            for platform in matrix_entry['platforms']:
                query_num = matrix_entry['query_id'].replace('Q', '').lstrip('0') or '1'
                guide_content += f"- `{platform}_query{query_num}_natural.json` (with {len(matrix_entry['natural_analysis_sections'])} JSON sections)\n"
                guide_content += f"- `{platform}_query{query_num}_controlled.json` (with {len(matrix_entry['controlled_analysis_sections'])} JSON sections)\n"
            
            guide_content += f"**Estimated Time**: {matrix_entry['estimated_time_minutes']} minutes (including JSON analysis)\n\n"
        
        # Add execution instructions with JSON requirements
        guide_content += f"""---

## 🚀 Execution Instructions

### Step 1: Setup
1. Create directory structure:
   ```
   stage2_execution/
   ├── natural_dataset/
   └── controlled_dataset/
   ```

2. Access the execution package JSON file for prompt templates
3. Ensure access to Pilot_mode_HM_V3-styled.txt specification for JSON requirements

### Step 2: Platform Testing Order
1. **Claude** (Anthropic) - Start here for baseline
2. **ChatGPT** (OpenAI) - Most widely used comparison  
3. **Gemini** (Google) - Strong search integration for controlled testing
4. **Grok** (X.AI) - Alternative perspective

### Step 3: For Each Query - CRITICAL JSON ANALYSIS REQUIREMENTS

#### 3A: Natural AI Testing
1. Use the Natural AI prompt (without search requirements)
2. Paste into each platform
3. **IMPORTANT**: The AI response must include BOTH sections:
   - `===== RESPONSE START =====` [AI's natural response] `===== RESPONSE END =====`
   - `===== ANALYSIS START =====` [Complete JSON analysis] `===== ANALYSIS END =====`
4. Save complete response as `{{platform}}_query{{number}}_natural.json`

#### 3B: Controlled AI Testing  
1. Use the Controlled AI prompt (with search verification requirements)
2. Paste into each platform
3. **IMPORTANT**: The AI response must include BOTH sections:
   - `===== RESPONSE START =====` [AI's verified response] `===== RESPONSE END =====`  
   - `===== ANALYSIS START =====` [Complete JSON analysis] `===== ANALYSIS END =====`
4. Save complete response as `{{platform}}_query{{number}}_controlled.json`

### Step 4: JSON Analysis Requirements

Each response file must contain:

#### Natural AI JSON Analysis (7 sections):
"""
        
        # Add natural AI analysis sections
        for section in json_guide['natural_ai_analysis_sections']:
            description = json_guide['natural_ai_analysis_sections'][section]
            guide_content += f"- **{section}**: {description}\n"
        
        guide_content += f"""
#### Controlled AI JSON Analysis (10 sections):
"""
        
        # Add controlled AI analysis sections  
        for section in json_guide['controlled_ai_analysis_sections']:
            description = json_guide['controlled_ai_analysis_sections'][section]
            guide_content += f"- **{section}**: {description}\n"
        
        guide_content += f"""

### Step 5: Data Quality Requirements
{json_guide['format_requirements']['response_section']}
{json_guide['format_requirements']['analysis_section']}
{json_guide['format_requirements']['json_validation']}
{json_guide['format_requirements']['completeness']}

**Critical**: {json_guide['data_quality_requirements']['no_placeholders']}

---

## ✅ Quality Control Checklist

### Pre-Execution
"""
        
        # Add quality checklist
        for check in execution_package['quality_control']['pre_execution_checks']:
            guide_content += f"- [ ] {check}\n"
        
        guide_content += "\n### During Execution\n"
        for check in execution_package['quality_control']['during_execution_checks']:
            guide_content += f"- [ ] {check}\n"
        
        guide_content += "\n### Post-Execution\n"
        for check in execution_package['quality_control']['post_execution_checks']:
            guide_content += f"- [ ] {check}\n"
        
        # Add JSON validation checklist
        guide_content += f"""
### JSON Analysis Validation
- [ ] All Natural AI responses include {len(json_guide['natural_ai_analysis_sections'])} JSON sections
- [ ] All Controlled AI responses include {len(json_guide['controlled_ai_analysis_sections'])} JSON sections  
- [ ] All JSON is valid and parseable
- [ ] No placeholder values remaining in analysis
- [ ] Quantitative data includes actual numbers, not placeholders
- [ ] Boolean values are true/false, not strings

---

## 📁 Expected File Structure

```
stage2_execution/
├── natural_dataset/          ({file_structure['stage2_execution']['subdirectories']['natural_dataset']['expected_files']} files with JSON analysis)
"""
        
        # Add natural dataset files
        for filename in file_structure['stage2_execution']['subdirectories']['natural_dataset']['file_list'][:5]:
            guide_content += f"│   ├── {filename}\n"
        
        if len(file_structure['stage2_execution']['subdirectories']['natural_dataset']['file_list']) > 5:
            remaining = len(file_structure['stage2_execution']['subdirectories']['natural_dataset']['file_list']) - 5
            guide_content += f"│   └── ... ({remaining} more files)\n"
        
        guide_content += f"└── controlled_dataset/       ({file_structure['stage2_execution']['subdirectories']['controlled_dataset']['expected_files']} files with JSON analysis)\n"
        
        # Add controlled dataset files
        for filename in file_structure['stage2_execution']['subdirectories']['controlled_dataset']['file_list'][:5]:
            guide_content += f"    ├── {filename}\n"
        
        if len(file_structure['stage2_execution']['subdirectories']['controlled_dataset']['file_list']) > 5:
            remaining = len(file_structure['stage2_execution']['subdirectories']['controlled_dataset']['file_list']) - 5
            guide_content += f"    └── ... ({remaining} more files)\n"
        
        guide_content += f"""

**Total Expected Files**: {file_structure['total_expected_files']} (all with complete JSON analysis)

---

## 🎉 Next Steps

After completing manual execution with JSON analysis:
1. Verify all {file_structure['total_expected_files']} files are created
2. Validate all JSON analysis sections are complete and valid
3. Run: `brandscope validate-responses {metadata.get('project_id', 'PROJECT_ID')}`
4. Proceed to Stage 3: Dual-Dataset Comparative Analysis
5. Generate strategic intelligence report

---

## ⚠️ Critical Reminders

1. **JSON Analysis is Mandatory**: Every response must include complete JSON analysis per specification
2. **No Shortcuts**: Both RESPONSE and ANALYSIS sections are required for every file
3. **Data Quality**: Replace all placeholders with actual analysis data
4. **Specification Compliance**: Follow Pilot_mode_HM_V3-styled.txt exactly
5. **File Naming**: Use exact naming convention for automated processing

---

*Generated by Brandscope AI Stage 2 Pipeline with JSON Analysis Requirements*  
*Specification: Pilot_mode_HM_V3-styled.txt*  
*Correlation ID: {metadata.get('correlation_id', 'Unknown')}*
"""
        
        return guide_content


# Integration function for Stage 2 prompt executor (UPDATED)
def generate_injectable_prompts(
    stage1_data: Stage1Data,
    priority_queries: List[Dict[str, Any]],
    customer_context: str,
    project_metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """
    Main function to generate injectable prompts with JSON analysis requirements.
    
    Args:
        stage1_data: Validated Stage 1 data
        priority_queries: Selected priority queries
        customer_context: Customer context string
        project_metadata: Project metadata
        
    Returns:
        Tuple of (execution_package, execution_guide_markdown) with JSON analysis requirements
    """
    # Generate execution package with JSON requirements
    package_generator = ExecutionPackageGenerator()
    execution_package = package_generator.generate_execution_package(
        stage1_data=stage1_data,
        priority_queries=priority_queries,
        customer_context=customer_context,
        project_metadata=project_metadata
    )
    
    # Generate execution guide with JSON requirements
    guide_generator = ExecutionGuideGenerator()
    execution_guide = guide_generator.generate_execution_guide(
        execution_package=execution_package,
        project_metadata=project_metadata
    )
    
    logger.info("Injectable prompts generated successfully with JSON analysis requirements",
               metadata={
                   "total_prompts": execution_package['metadata']['prompts_generated'],
                   "guide_length": len(execution_guide),
                   "json_analysis_included": True,
                   "specification_compliance": execution_package['metadata']['specification_compliance']
               })
    
    return execution_package, execution_guide