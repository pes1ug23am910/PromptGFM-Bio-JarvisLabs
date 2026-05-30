# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of PromptGFM-Bio seriously. If you have discovered a security vulnerability, please report it to us privately.

### Where to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, report them via one of these methods:

1. **Email**: yashverma.pes@gmail.com
2. **GitHub Security Advisory**: Use the "Security" tab on our GitHub repository

### What to Include

Please include the following information:

- **Type of vulnerability**: (e.g., code injection, XSS, data leak)
- **Affected versions**: Which versions are affected
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Potential impact**: What an attacker could potentially do
- **Suggested fix** (if known): Any ideas for how to fix it

### Response Timeline

- **Initial response**: Within 48 hours of receipt
- **Status update**: Within 7 days with assessment
- **Fix timeline**: Depends on severity
  - **Critical**: Within 24-48 hours
  - **High**: Within 1 week
  - **Medium**: Within 2 weeks
  - **Low**: Within 30 days

### Disclosure Policy

- We will coordinate disclosure with you
- We will publicly acknowledge your contribution (if desired)
- We will release a fix before any public disclosure
- We will credit you in our release notes

## Security Best Practices

### For Users

1. **Keep dependencies updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use virtual environments**
   ```bash
   conda create -n promptgfm python=3.10
   ```

3. **Don't commit sensitive data**
   - Never commit API keys, passwords, or credentials
   - Use `.env` files (excluded by `.gitignore`)
   - Use environment variables for secrets

4. **Verify data sources**
   - Only download data from official sources
   - Verify checksums when available
   - Be cautious of user-provided input

5. **Run in isolated environments**
   - Use Docker containers for deployment
   - Limit file system access
   - Use non-root users

### For Developers

1. **Code Security**
   - Validate all inputs
   - Sanitize user-provided text
   - Avoid eval() and exec()
   - Use parameterized queries (if using databases)

2. **Dependency Management**
   - Regularly update dependencies
   - Use `pip-audit` or `safety` to check for vulnerabilities
   ```bash
   pip install safety
   safety check -r requirements.txt
   ```

3. **Secrets Management**
   - Use environment variables
   - Never hardcode credentials
   - Use `.env.example` as template
   - Rotate API keys regularly

4. **Data Privacy**
   - No PHI (Protected Health Information)
   - Anonymize any logs
   - Follow GDPR/HIPAA if applicable
   - Respect data source licenses

## Known Security Considerations

### Model Security

1. **Adversarial Inputs**
   - Disease descriptions are processed by text models
   - Malicious inputs could potentially cause issues
   - Validate and sanitize all text inputs

2. **Model Poisoning**
   - Ensure training data comes from trusted sources
   - Verify data integrity
   - Monitor for anomalous predictions

3. **Privacy**
   - No patient data should be used
   - De-identify any real clinical data
   - Aggregate statistics only

### Dependency Vulnerabilities

We regularly scan for vulnerabilities in:
- PyTorch and related libraries
- Transformers (HuggingFace)
- NumPy, Pandas
- Web frameworks (if API deployed)

Run security scans:
```bash
# Install tools
pip install bandit safety

# Scan code
bandit -r src/

# Check dependencies
safety check -r requirements.txt
```

### API Security (if deployed)

1. **Authentication**
   - Implement API key authentication
   - Use JWT tokens for session management
   - Rate limiting to prevent abuse

2. **Input Validation**
   - Validate all API inputs
   - Set maximum input lengths
   - Sanitize disease descriptions

3. **HTTPS**
   - Always use HTTPS in production
   - Use valid SSL/TLS certificates
   - Enable HSTS headers

4. **Rate Limiting**
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/predict")
   @limiter.limit("10/minute")
   async def predict(...):
       ...
   ```

## Secure Configuration

### Production Deployment Checklist

- [ ] All secrets in environment variables (not in code)
- [ ] HTTPS enabled
- [ ] Authentication implemented
- [ ] Rate limiting configured
- [ ] Input validation in place
- [ ] Logging configured (without sensitive data)
- [ ] Error messages don't leak information
- [ ] Dependencies up to date
- [ ] Security headers configured
- [ ] Regular backups configured

### Example Secure Configuration

```yaml
# Example only: create a deployment-specific config in your environment.
# Suggested base: copy configs/workstation_config.yaml and extend it with security settings.
security:
  enable_authentication: true
  rate_limit:
    enabled: true
    requests_per_minute: 60
  input_validation:
    max_description_length: 1000
    allowed_characters: "alphanumeric_and_punctuation"
  logging:
    level: "INFO"
    sanitize_inputs: true
    exclude_fields: ["api_key", "user_id"]
```

## Data Security

### Handling Biomedical Data

1. **Public Data Only**
   - Use only publicly available databases
   - No clinical trial data without proper authorization
   - No patient-level data

2. **Data Provenance**
   - Document data sources
   - Verify data licenses
   - Maintain chain of trust

3. **Storage**
   - Encrypt sensitive data at rest
   - Use secure cloud storage
   - Regular backups with encryption

## Incident Response

If you discover you've been compromised:

1. **Immediate Actions**
   - Disconnect affected systems
   - Rotate all credentials
   - Notify all users if data breach

2. **Investigation**
   - Determine scope of breach
   - Identify compromised data
   - Document timeline

3. **Remediation**
   - Apply security patches
   - Update configurations
   - Improve monitoring

4. **Communication**
   - Notify affected parties
   - Public disclosure (if appropriate)
   - Lessons learned

## Security Updates

Security updates will be announced:
- GitHub Security Advisories
- Release notes (for public issues)
- Email to registered users (for critical issues)

Subscribe to notifications:
- Watch the GitHub repository
- Enable security alerts
- Join mailing list (if available)

## Compliance

This project aims to be compliant with:
- **GDPR**: No personal data collection
- **HIPAA**: Not intended for clinical use
- **SOC 2**: Security best practices
- **Open Source Security Foundation (OpenSSF)**: Best practices

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [PyTorch Security](https://pytorch.org/docs/stable/notes/security.html)

## Questions?

For security-related questions (non-vulnerability):
- GitHub Discussions: [Link to discussions]
- Email: [general-contact@example.com]

**For vulnerabilities, use the private reporting methods above.**

---

Last updated: 2026-03-07
Maintainer: Yash Verma (PES1UG23AM910)
