# HR Documents - Data Explainer

## Purpose

This folder contains the source documents used as the knowledge base for the AI Employee Onboarding Assistant.

The prototype uses simulated HR and IT documents instead of real company documents. This makes the project safer to develop and evaluate because no confidential employee information is exposed.

## Documents

### Annual_Leave_Policy.txt

Contains information about:

Annual leave entitlement

Leave application timing

Carry-forward rules

Manager approval

Pro-rated leave for new employees

Expiry of unused leave

### Employee_Benefits_Guide.txt

Contains information about:

Dental benefits

Wellness allowance

Health screening

Medical insurance

Dependent coverage

Benefit support contacts

### IT_Support_Guide.txt

Contains information about:

Password reset

IT Help Desk availability

Company email access

Laptop support

Software installation

Cybersecurity reporting

## Why Simulated Data Was Used

Real HR documents may contain confidential or sensitive company information.

The prototype therefore uses simulated documents so that:

The project can be shared publicly on GitHub

The evaluation can be reproduced

No employee personal data is exposed

The system can be tested without company access restrictions

## Data Scope

The current dataset is intentionally small.

It contains only three document categories:

Leave

Benefits

IT Support

This is enough to test the complete RAG workflow, but it does not represent the complexity of a real enterprise knowledge base.

A real company may have:

Hundreds of documents

Different document versions

PDF files and tables

Conflicting policies

Department-specific rules

Access restrictions

Outdated information

## Privacy Considerations

The prototype does not require or store personal employee information such as:

Employee names

Home addresses

Identification numbers

Salary information

Medical records

Phone numbers

For a production deployment, additional controls would be required, including:

Authentication

Role-based access control

Encryption

Audit logging

Document permissions

Data retention rules

Data residency review

Document version management

Only approved company documents should be added to the production retrieval index.

## Limitation

Because the current dataset is small and simulated, the evaluation results should be interpreted as prototype-level results.

High accuracy on these documents does not guarantee the same performance on a large real-world company knowledge base.
