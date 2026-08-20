# Umbrella AI Research

Umbrella – Frontend & Database Structure Prompt

I want you to design the frontend architecture and local data structure for my AI platform called Umbrella.

Project Overview

Umbrella is an AI-powered research platform that orchestrates multiple specialized AI agents to assist researchers in genomics, biodiversity, evolution, trait discovery, protein visualization, and scientific literature analysis. The platform provides a conversational interface where users interact with an intelligent multi-agent system capable of planning, reasoning, and coordinating multiple AI agents.

For this phase, do NOT implement real authentication or connect to any backend/database service. Authentication should be UI only, using mock data and local state. The goal is to build the complete user experience before integrating a real backend.

Introduction (Pre-Authentication Landing)

Before the authentication flow, create a landing / introduction screen that explains what Umbrella is and what it does.

This page should feel like a premium AI product landing experience and should appear before Sign In / Sign Up.

Content to Include

Hero Section

Product name: Umbrella

Tagline: “An AI-powered research ecosystem for genomics, biodiversity, and scientific discovery.”

Short description:
Explain that Umbrella is a multi-agent AI platform that helps researchers analyze biological data, explore scientific literature, and generate insights across genomics, evolution, traits, and protein structures.

Key Capabilities Section

Show a clean overview of what the platform does:

Genome Analysis & Reconstruction

Biodiversity Intelligence

Evolutionary Research Support

Trait Discovery

Protein Visualization

Scientific Literature Exploration

Multi-Agent AI Orchestration

How It Works (Simple Explanation)

Briefly explain:

Users interact with a conversational AI interface

Multiple specialized AI agents collaborate behind the scenes

The system plans, reasons, and executes research tasks automatically

Call to Action

Include two primary buttons:

Sign In

Get Started

These should lead into the authentication flow.

Authentication Flow (UI Only)

Create modern Sign In and Sign Up pages.

Sign Up

The registration process should feel like onboarding rather than a traditional signup form.

Step 1 — Basic Information

Collect:

Full Name

Email Address

Password

Role

Available roles:

Student

Researcher

Professor

Conservation Scientist

Bioinformatician

Developer

Other

Step 2 — User Goals

Ask onboarding questions such as:

Why are you using Umbrella?

What is your main research interest?

What are you hoping to accomplish?

Which AI capabilities are you most interested in?

Examples:

Genome Reconstruction

Biodiversity Analysis

Evolutionary Studies

Trait Discovery

Protein Visualization

Scientific Literature

Species Identification

Store these responses only in local mock state.

Do not implement real authentication or database persistence.

Main Application Layout

After onboarding, the user enters the main platform.

The design should be modern, minimal, and research-oriented, similar to ChatGPT or Claude, but adapted for a multi-agent AI ecosystem.

The interface consists of:

Left Sidebar

Contains:

Conversation History

Create New Conversation

Search Conversations

Rename Conversation

Delete Conversation

The sidebar should be collapsible.

Main Conversation Area

Contains:

Conversation messages

Markdown rendering

Code blocks

Scientific formatting

Smooth scrolling

Typing animation

Multi-Agent Thinking Panel

Above the user input, add a compact Agent Thinking & Planning section.

This panel should:

Show which agent(s) are currently active

Display planning steps

Display orchestration progress

Show reasoning status

Show task execution timeline

The panel should be:

Collapsed by default

Expandable with a smooth animation

Non-intrusive so it doesn't distract from the conversation

Example:

Agent Thinking ▼

Planning workflow...

Selecting Genome Agent...

Requesting Literature Agent...

Combining responses...

Finalizing answer...

This section will later display real orchestration events.

User Input Area

The bottom input area should include:

Large text input

Send button

Attachment button

Optional microphone button (placeholder only)

Clean, modern, and centered.

Appearance

Support both:

Light Mode

Dark Mode

The user should be able to switch themes instantly.

Color Palette

Primary Colors:

Red (#C1121F or similar)

White

Dark Gray / Black

The design should communicate:

Scientific

Modern

Intelligent

Premium

Minimal

Avoid excessive colors.

Components

Create reusable components for:

Buttons

Cards

Inputs

Sidebar

Conversation List

Chat Messages

Agent Thinking Panel

User Avatar

Theme Toggle

Loading Indicators

Empty States

Mock Data Structure

Create local mock models (no real database) for:

User

id

name

email

role

purpose

researchInterests

createdAt

Conversation

id

title

createdAt

updatedAt

Message

id

conversationId

sender

content

timestamp

Agent Activity

id

conversationId

agentName

status

description

timestamp

Use mock JSON or local state to populate the interface.

Important Constraints

Do NOT integrate Supabase, Firebase, PostgreSQL, or any backend.

Do NOT implement real login or authentication.

Do NOT connect to APIs.

Simulate all data using local mock state.

Focus entirely on building a polished, responsive UI and the application structure.

The final result should feel like a professional AI research platform ready to be connected to a backend in a future development phase.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/2e5ac791-1aad-442f-a50b-94b5bd46b383).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
