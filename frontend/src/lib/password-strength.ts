export type PasswordStrengthScore = 0 | 1 | 2 | 3 | 4;

const COMMON_PASSWORDS = new Set([
  "123456789012",
  "admin123456",
  "letmein123456",
  "password",
  "password123",
  "password1234",
  "qwerty123456",
  "umbrella1234",
  "welcome123456",
]);

const PERSONAL_IDENTIFIER_MIN_LENGTH = 4;

const LABELS: Record<PasswordStrengthScore, string> = {
  0: "Too short",
  1: "Weak",
  2: "Fair",
  3: "Good",
  4: "Strong",
};

function comparisonKey(value: string): string {
  return [...value.normalize("NFKD").toLocaleLowerCase("en-US")]
    .filter((character) => /[\p{L}\p{N}]/u.test(character))
    .join("");
}

function personalIdentifiers(email: string, fullName: string): Set<string> {
  const emailIdentifier = email.trim().toLocaleLowerCase("en-US").split("@", 1)[0] ?? "";
  const candidates = [
    emailIdentifier,
    ...emailIdentifier.split(/[^\p{L}\p{N}]+/u),
    fullName,
    ...fullName.split(/[^\p{L}\p{N}]+/u),
  ];

  return new Set(
    candidates
      .map(comparisonKey)
      .filter((identifier) => identifier.length >= PERSONAL_IDENTIFIER_MIN_LENGTH),
  );
}

export interface PasswordPolicyResult {
  valid: boolean;
  checks: {
    length: boolean;
    notCommon: boolean;
    noPersonalInfo: boolean;
  };
  message: string | null;
}

export function evaluatePasswordPolicy(
  password: string,
  email = "",
  fullName = "",
): PasswordPolicyResult {
  const length = password.length >= 12 && password.length <= 128;
  const notCommon = !COMMON_PASSWORDS.has(password.trim().toLocaleLowerCase("en-US"));
  const passwordKey = comparisonKey(password);
  const noPersonalInfo = ![...personalIdentifiers(email, fullName)].some((identifier) =>
    passwordKey.includes(identifier),
  );
  const message = !length
    ? "Use between 12 and 128 characters."
    : !notCommon
      ? "Choose a less common password."
      : !noPersonalInfo
        ? "Do not include your name or email identifier."
        : null;

  return {
    valid: length && notCommon && noPersonalInfo,
    checks: { length, notCommon, noPersonalInfo },
    message,
  };
}

export function getPasswordStrength(password: string): {
  score: PasswordStrengthScore;
  label: string;
} {
  if (password.length === 0) return { score: 0, label: "" };
  if (password.length < 12) return { score: 0, label: LABELS[0] };

  let score = 1;
  if (password.length >= 16) score++;
  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
  if (/\d/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;

  const clamped = Math.min(score, 4) as PasswordStrengthScore;
  return { score: clamped, label: LABELS[clamped] };
}
