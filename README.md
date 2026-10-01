# AI Employee Onboarding Assistant

## Project Overview

The AI Employee Onboarding Assistant is a RAG-based chatbot designed to help new employees quickly find information about annual leave, employee benefits, and IT support.

Instead of searching through several HR documents or contacting HR for basic questions, employees can ask questions in natural language. The system retrieves the most relevant approved HR document and uses an LLM to generate a short answer based on that document.

If the available documents do not contain enough information, the system does not guess. It asks the employee to contact Human HR.

---

## Persona

### Primary User: New Employee

A new employee may need quick answers about:

Annual leave

Employee benefits

Health screening

Medical insurance

Password reset

Laptop support

Software installation

IT security

### Secondary Users: HR and IT Support Teams

HR and IT teams can use the assistant to reduce repeated basic questions and spend more time on complex employee issues.

---

## Problem

Company policies and support information may already exist in internal documents, but employees may not know:

Which document contains the answer

Which keywords to search

How the company describes a policy

For example, an employee may ask:

> How much holiday can I take each year?

while the company document uses the term:

> annual leave

A simple keyword search may therefore fail when different words have similar meanings.

---

## Input

The main input is a natural-language employee question.

Example:

```text
How much holiday can I take each year?

