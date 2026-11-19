# Budget Alarms & Cost Guardrails
# Ensures strictly $0 unexpected charges by alerting at early spending thresholds ($1)

resource "google_billing_budget" "zero_dollar_guardrail" {
  count           = var.billing_account_id != "" ? 1 : 0
  billing_account = var.billing_account_id
  display_name    = "HSCM Free-Tier Zero Dollar Guardrail"

  budget_filter {
    projects = ["projects/${var.gcp_project_id}"]
    credit_types_treatment = "INCLUDE_ALL_CREDITS"
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = "1" # $1 limit alert
    }
  }

  threshold_rules {
    threshold_percent = 0.50 # Alert at $0.50
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 0.90 # Alert at $0.90
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 1.00 # Alert at $1.00
    spend_basis       = "CURRENT_SPEND"
  }
}
