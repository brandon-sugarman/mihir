from pydantic import BaseModel, Field, create_model
from typing import List, Type, Dict, Any, Tuple, Literal, Optional


PYDANTIC_CHUNK_SIZE = 50

class k1_cover_page(BaseModel):
    partnership_name: str
    partnership_employer_identification_number: str

    line_1_ordinary_business_income_loss_passive: int
    line_1_ordinary_business_income_loss: int
    line_2_net_rental_real_estate_income_loss: int
    line_3_other_rental_income_loss: int
    line_4a_guaranteed_payments_for_services: int
    line_4b_guaranteed_payments_for_capital: int
    line_4c_total_guaranteed_payments: int
    line_5_interest_income: int
    line_5_interest_income_us_government_interest: int
    line_6a_ordinary_dividends: int
    line_6b_qualified_dividends: int
    line_6c_dividend_equivalents: int
    line_7_royalties: int
    line_8_net_short_term_capital_gain_loss: int
    line_9a_net_long_term_capital_gain_loss: int
    line_9b_collectibles_28_percent_gain_loss: int
    line_9c_uncaptured_section_1250_gain: int
    line_10_net_section_1231_gain_loss: int
    line_12_section_179_deduction: int
    line_18a_tax_exempt_interest_income: int
    line_18b_other_tax_exempt_income: int
    line_18c_nondeductible_expenses: int

    line_21_foreign_taxes_paid_or_accrued: int

    line_13m_amounts_paid_for_medical_insurance: int

    capital_contributions_during_year: int
    other_increase_decrease_income_items: int
    withdrawals_and_distributions_cash: int
    ending_capital_account: int

class k1_federal_footnotes(BaseModel):
    line_9a_net_long_term_capital_gain_loss_property_held_3_years_or_less_logic: str
    line_9a_net_long_term_capital_gain_loss_property_held_3_years_or_less: int
    line_9a_net_long_term_capital_gain_loss_property_held_more_than_3_years_logic: str
    line_9a_net_long_term_capital_gain_loss_property_held_more_than_3_years: int
    line_11a_other_income_total: int
    line_11b_involuntary_conversions: int
    line_11c_section_1256_gain_loss_logic: str
    line_11c_section_1256_gain_loss: int
    line_11d_mining_exploration_costs_recapture: int
    line_11e_cancellation_of_debt: int
    line_11f_section_743b_positive_adjustments: int
    line_11h_section_951a_inclusion: int
    line_11i_gain_loss_from_disposition_of_oil_gas_geothermal_mineral_properties: int
    line_11j_recovery_of_tax_benefit_items: int
    line_11k_gambling_gains_losses: int
    line_11l_any_income_gain_loss_to_partnership_under_distribution_under_751b: int
    line_11m_gain_eligible_for_section_1045_rollover_purchased_partnership_short_term: int
    line_11m_gain_eligible_for_section_1045_rollover_purchased_partnership_long_term: int
    line_11n_gain_eligible_for_section_1045_rollover_not_purchased_partnership_short_term: int
    line_11n_gain_eligible_for_section_1045_rollover_not_purchased_partnership_long_term: int
    line_11o_sale_or_exchange_of_qsb_stock_with_section_1202_exclusion_short_term: int
    line_11o_sale_or_exchange_of_qsb_stock_with_section_1202_exclusion_long_term: int
    line_11p_gain_or_loss_on_disposition_of_farm_recapture_property_and_other_items_to_which_section_1252_applies_short_term: int
    line_11q_gain_or_loss_on_fannie_mae_or_freddie_mac_qualified_preferred_stock: int
    line_11r_specially_allocated_ordinary_gain_loss: int
    line_11s_non_portfolio_gain_loss_stcg: int
    line_11s_non_portfolio_gain_loss_ltcg: int
    line_11ZZ_from_pass_through_entities_other_income_loss: int
    line_11ZZ_income_from_depletion_properties: int
    line_11ZZ_gain_loss_capital_net_long_term: int
    line_11ZZ_gain_loss_capital_net_long_term_qsbs: int
    line_11ZZ_gain_loss_capital_net_short_term: int
    line_11ZZ_gain_loss_capital_sale_of_pfic_long_term: int
    line_11ZZ_gain_loss_ordinary_from_form_4797: int
    line_11ZZ_foreign_futures_trading_gain_loss: int
    line_11ZZ_interest_income: int
    line_11ZZ_interest_income_self_charged_interest: int
    line_11ZZ_interest_income_us_government: int
    line_11ZZ_interest_income_trader_expense: int
    line_11ZZ_mtm_income_loss: int
    line_11ZZ_ordinary_income_section_475f: int
    line_11ZZ_other_income_loss: int
    line_11ZZ_other_portfolio_income_loss: int
    line_11ZZ_other_trade_business_expense: int
    line_11ZZ_other_trade_business_income: int
    line_11ZZ_pfic_1291_excess_distributions: int
    line_11ZZ_pfic_qef_income: int
    line_11ZZ_pfic_qef_income_section_1250_gain: int
    line_11ZZ_section_965_income: int
    line_11ZZ_section_986_total: int
    line_11ZZ_section_987_total: int
    line_11zz_section_988_total_logic: str
    line_11ZZ_section_988_total: int
    line_11ZZ_swap_net_income_loss: int
    line_11ZZ_divedends_equivalent_swap_income_total: int
    line_11ZZ_other_ordinary_income_loss_total: int
    line_11ZZ_interest_income_domestic: int
    line_11ZZ_interest_income_foreign: int
    line_11ZZ_dividends_qualified_domestic: int
    line_11ZZ_dividends_qualified_foreign: int
    line_11ZZ_dividends_non_qualified_domestic: int
    line_11ZZ_dividends_non_qualified_foreign: int
    line_11ZZ_operating_expense: int
    line_11ZZ_business_interest_expense: int
    line_11ZZ_ptp_ordinary_income: int

    line_13a_cash_contributions_50_percent: int
    line_13b_cash_contributions_30_percent: int
    line_13g_cash_contributions_100_percent: int
    line_13c_non_cash_contributions_50_percent: int
    line_13d_non_cash_contributions_30_percent: int
    line_13e_capital_gain_property_to_50_percent_organization_30_percent: int
    line_13f_capital_gain_property_20_percent: int
    line_13g_non_cash_contributions_qualified_conservation_100_percent: int
    
    line_13h_investment_interest_investing_schedule_A_logic: str
    line_13h_investment_interest_investing_schedule_A: int
    line_13h_investment_interest_trading_schedule_E_logic: str
    line_13h_investment_interest_trading_schedule_E: int
    
    line_13i_royalty_deductions: int
    line_13j_section_59_e_2_expenditures: int
    line_13k_excess_business_interest_expense: int
    line_13l_deductions_portfolio_other_logic: str
    line_13l_deductions_portfolio_other: int
    line_13n_educational_assistance_benefits: int
    line_13o_dependent_care_benefits: int
    line_13p_preproductive_period_expenses: int
    line_13r_pension_and_iras: int
    line_13s_reforestation_expense_deduction: int
    line_13v_section_743b_negative_adjustments: int
    line_13w_soil_and_water_conservation: int
    line_13x_film_television_and_theatrical_production_expenditures: int
    line_13y_expenditures_for_removal_of_barriers: int
    
    line_13z_itemized_deductions_total: int
    line_13z_total: int
    
    
    line_13AA_contributions_to_a_capital_construction_fund: int
    line_13AB_penalty_on_early_withdrawal_of_savings: int
    line_13AC_interest_expense_allocated_to_debt_financed_distributions: int
    line_13AD_interest_expense_on_working_interest_in_oil_or_gas: int
   
    line_13AE_deductions_portfolio_income_logic: str
    line_13AE_deductions_portfolio_income: int
   
    line_13ZZ_other_deductions_total_logic: str
    line_13ZZ_other_deductions_total: int
  
  
    line_14_net_earnings_loss_from_self_employment: int
    line_15e_qualified_rehabilitation_expenditures: int
    line_15f_other_rental_real_estate_credits: int
    line_15g_other_rental_credits: int
    line_15h_undistributed_capital_gains_credit: int
    line_15i_biofuel_producer_credit: int
    line_15j_work_opportunity_credit: int
    line_15l_empowerment_zone_employment_credit: int
    line_15m_credit_for_increasing_research_activities: int
    line_15n_credit_for_employer_social_security_and_medicare_taxes: int
    line_15o_backup_withholding: int
    line_15v_advanced_manufacturing_production_credit: int
    line_15y_clean_hydrogen_production_credit: int
    line_15aa_enhanced_oil_recovery_credit: int
    line_15ab_renewable_electricity_production_credit: int
    line_15zz_other_credits: int
    line_15_small_employer_auto_enrollment_credit_form_8881: int

    line_15_aviation_fuels_form_8864: int
    line_15_reserved: int
    line_15_alternative_fuel_vehicle_refueling_property_form_8911: int
    line_15_alternative_motor_vehicle: int

    line_15_alternative_motor_vehicle_refueling_property_form_8911: int   
    line_15_biodiesel_and_renewable_diesel_fuels_form_8864: int
    line_15_build_america_bond: int
    line_15_carbon_oxide_sequestration: int
    line_15_clean_renewable_energy_bond: int
    line_15_disabled_access: int
    line_15_distilled_spirits: int
    line_15_electricity_closed_loop_biomass: int
    line_15_electricity_open_loop_biomass: int
    line_15_employer_provided_childcare_facilities: int
    line_15_empowerment_zone_employment_form_8844: int
    line_15_increasing_research_eligible_small_business: int
    line_15_increase_research_activities_form_6765: int
    line_15_indian_coal_production_facility: int
    line_15_indian_employment_form_8845: int
    line_15_low_sulfur_diesel_fuel_production: int
    line_15_military_spouse_participation: int
    line_15_qualified_commercial_clean_vehicle: int
    line_15_lih_section_42_j_5: int
    line_15_lih_from_other_partnerships: int
    line_15_new_markets: int
    line_15_employer_credit_for_paid_family_and_medical_leave_form_8994: int
    line_15_employee_retention_credit: int
    line_15_new_clean_renewable_energy_bond: int
    line_15_orphan_drug: int
    line_15_qual_energy_conservation_energy_bond: int
    line_15_new_clean_vehicle_business_investment_use: int
    line_15_qualified_railroad_track_maintenance_form_8900: int
    line_15_qual_school_construction_bond: int
    line_15_refined_coal_not_produced_in_4_year_period: int
    line_15_small_employer_health_insurance_premiums: int
    line_15_small_employer_pension_plan_start_up: int
    line_15_oil_and_gas_production_from_marginal_wells_form_8904: int
    line_15_taxable_income_attributable_to_pass_through: int
    line_15_work_opportunity_credit: int
    line_15_energy_efficient_home_credit: int
    line_15_mine_rescue_team_training_form_8923: int
    line_15_employer_differential_wage_payments_8932: int
    line_17a_post_1986_depreciation_adjustment: int
    line_17b_adjusted_gain_or_loss: int
    line_17c_depletion_other_than_oil_gas: int
    line_17d_oil_gas_geothermal_mineral_gross_income: int
    line_17e_oil_gas_geothermal_mineral_deductions: int
    line_17f_other_amt_items: int
    line_20_net_irc_section_988_gross_losses: int
    line_20AA_section_704c_information: int
    line_20AB_section_751_gain_loss: int
    line_20AD_deemed_section_1250_unrecaptured_gain: int
    line_20AG_gross_receipts_section_448_c: int
    line_20_installment_sale_deferred_gain_capital_gain: int
    line_20_installment_sale_outstanding_obligations: int
    line_20_installment_sale_deferred_gain_interest: int
    line_20N_interest_expense_for_corporate_partners: int
    line_20V_unrelated_business_taxable_income_logic: str
    line_20V_unrelated_business_taxable_income: int
    line_20AE_excess_taxable_income:  int
    line_20AF_excess_business_interest_income:  int
    line_20AM_section_1061_information: int
    line_20O_453I3_information: int
    line_20P_452Ac_information: int


def create_chunked_models(
    models: List[Type[BaseModel]], 
    chunk_size: int = PYDANTIC_CHUNK_SIZE
) -> List[Type[BaseModel]]:
    chunked_models = []
    
    for model in models:
        if len(model.model_fields) <= chunk_size:
            chunked_models.append(model)
            continue
            
        current_fields: Dict[str, Tuple[Any, Any]] = {}
        count = 0
        chunk_count = 0
        current_model_name = model.__name__
        
        for field_name, field in model.model_fields.items():
            current_fields[field_name] = (field.annotation, field)
            count += 1
            
            if count >= chunk_size or field_name == list(model.model_fields.keys())[-1]:
                chunk_name = f"{current_model_name}_Chunk_{chunk_count}"
                ChunkModel = create_model(chunk_name, **{
                    name: (annotation, field)
                        for name, (annotation, field) in current_fields.items()
                })
                chunked_models.append(ChunkModel)
                current_fields = {}
                count = 0
                chunk_count += 1
    
    return chunked_models


k1_pydantic_classes = create_chunked_models(
    models=[
        k1_cover_page, 
        k1_federal_footnotes
    ],
    chunk_size=PYDANTIC_CHUNK_SIZE
)