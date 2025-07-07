# Airline Agent Backend

A FastAPI-based airline customer service agent system using Groq AI models and Supabase database.

## Features

- **Multi-Agent System**: Triage, FAQ, Seat Booking, Flight Status, and Cancellation agents
- **Real-time Data**: Integrates with Supabase for customer, flight, and booking data
- **AI-Powered**: Uses Groq's Llama models for natural language processing
- **Guardrails**: Built-in relevance and jailbreak detection
- **RESTful API**: FastAPI endpoints for frontend integration

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

   Required variables:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_ANON_KEY`: Your Supabase anonymous key
   - `GROQ_API_KEY`: Your Groq API key

3. **Database Setup**:
   Run the SQL scripts in your Supabase dashboard to create the required tables:
   - `customers` table
   - `flights` table
   - `bookings` table
   - `conversations` table

4. **Run the Server**:
   ```bash
   python -m uvicorn api:app --reload --port 8000
   ```

## API Endpoints

- `GET /user/{account_number}`: Login and get customer details
- `POST /chat`: Main chat endpoint for agent interactions
- `GET /health`: Health check endpoint

## Database Schema

### Customers Table
```sql
CREATE TABLE public.customers (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    account_number text UNIQUE NOT NULL,
    name text NOT NULL,
    email text
);
```

### Flights Table
```sql
CREATE TABLE public.flights (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    flight_number text UNIQUE NOT NULL,
    origin text NOT NULL,
    destination text NOT NULL,
    scheduled_departure timestamptz NOT NULL,
    scheduled_arrival timestamptz NOT NULL,
    current_status text NOT NULL,
    gate text,
    terminal text,
    delay_minutes integer
);
```

### Bookings Table
```sql
CREATE TABLE public.bookings (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    confirmation_number text UNIQUE NOT NULL,
    flight_id uuid NOT NULL REFERENCES public.flights(id),
    customer_id uuid NOT NULL REFERENCES public.customers(id),
    seat_number text,
    booking_status text NOT NULL
);
```

### Conversations Table
```sql
CREATE TABLE public.conversations (
    session_id text NOT NULL PRIMARY KEY,
    history jsonb NULL,
    context jsonb NULL,
    current_agent text NULL,
    last_updated timestamp with time zone NOT NULL DEFAULT now()
);
```

## Agent Architecture

1. **Triage Agent**: Main entry point, routes to specialists
2. **FAQ Agent**: Handles general airline policy questions
3. **Seat Booking Agent**: Manages seat changes and seat map display
4. **Flight Status Agent**: Provides real-time flight information
5. **Cancellation Agent**: Processes flight cancellations

## Sample Data

Use the provided sample data in your Supabase dashboard:
- Customer accounts: CUST001, CUST002, CUST003, etc.
- Flight numbers: FLT-100, AA2025, AI500, etc.
- Confirmation numbers: ABC100, XYZ999, PQR777, etc.

## Development

The system is designed to be modular and extensible. Each agent has specific responsibilities and can hand off to other agents as needed.

### Adding New Agents

1. Create agent instructions function
2. Define any required tools
3. Add handoff logic
4. Register with triage agent

### Adding New Tools

1. Use `@function_tool` decorator
2. Implement database operations in `database.py`
3. Add to appropriate agent's tools list

## Troubleshooting

- Ensure all environment variables are set correctly
- Check Supabase connection and table structure
- Verify Groq API key is valid
- Check logs for detailed error messages