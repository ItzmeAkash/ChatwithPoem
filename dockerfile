# Start from the base image
FROM ankane/pgvector

# Set environment variables
ENV POSTGRES_PASSWORD=admin

# Expose the default PostgreSQL port
EXPOSE 5432

# Default command to run PostgreSQL
CMD ["postgres"]
