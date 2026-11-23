# Claim and Policy Workflows

## Claim Workflow

A claim starts in the SUBMITTED state. A SUBMITTED claim can move to UNDER_REVIEW or WITHDRAWN. An UNDER_REVIEW claim can move to INFO_REQUESTED, APPROVED, DENIED or WITHDRAWN. An INFO_REQUESTED claim can move back to UNDER_REVIEW. An APPROVED claim can move to PAID. A DENIED claim can move to APPEALED, and an APPEALED claim moves back to UNDER_REVIEW. PAID and WITHDRAWN are final states.

## Policy Workflow

A policy starts in the QUOTED state and moves to ISSUED when the application is accepted. An ISSUED policy moves to IN_FORCE when the first premium is paid. An IN_FORCE policy can move to GRACE or CANCELLED. A GRACE policy returns to IN_FORCE when the premium is paid or moves to LAPSED when the grace period ends. A LAPSED policy can move to REINSTATEMENT_REVIEW. A policy in REINSTATEMENT_REVIEW moves to IN_FORCE if approved or returns to LAPSED if declined. CANCELLED is a final state.

## Requests Through the Assistant

Every state change requested through the assistant needs a reviewer's approval before it is applied. A reviewer cannot approve a request that they submitted themselves. Available forms are the first notice of loss (fnol), address change (address_change), beneficiary change (beneficiary_change) and cancellation request (cancellation_request). A first notice of loss form needs the policy number, loss date, loss type, description and estimated amount.
