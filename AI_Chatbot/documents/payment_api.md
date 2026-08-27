# Payment API Documentation

## Overview

The Payment API allows applications to create and manage customer payments.

## Authentication

The API uses **Bearer Token authentication**.

Clients must include the token in the HTTP Authorization header:

    Authorization: Bearer YOUR_API_TOKEN

## Create Payment

Endpoint:

    POST /api/v1/payments

Required parameters:

- `amount` - Payment amount in USD
- `currency` - Three-letter currency code
- `customer_id` - Unique customer identifier

## Payment Limits

The minimum payment amount is **$1.00**.

The maximum payment amount is **$10,000.00 per transaction**.

## Rate Limits

Each API key can make up to **100 requests per minute**.