from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    proxy_api_key: SecretStr
    # Bedrock API key, sent as `Authorization: Bearer`; uses AWS's standard variable name.
    bedrock_api_key: SecretStr = Field(validation_alias="AWS_BEARER_TOKEN_BEDROCK")
    aws_region: str = "us-east-1"
    bedrock_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    bedrock_base_url: str | None = None  # override for VPC endpoints / tests
    default_max_tokens: int = 1024  # Converse needs a limit; OpenAI clients may omit it
    pii_enabled: bool = False  # needs the `pii` extra (gliner)
    pii_model: str = "nvidia/gliner-PII"
    pii_threshold: float = 0.4
    connect_timeout: float = 5.0
    read_timeout: float = 120.0

    @property
    def base_url(self) -> str:
        return (self.bedrock_base_url or f"https://bedrock-runtime.{self.aws_region}.amazonaws.com").rstrip("/")
