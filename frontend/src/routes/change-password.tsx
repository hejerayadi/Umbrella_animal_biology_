import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AuthShell } from "@/components/umbrella/auth-shell";
import { resolveAuthDestination, useAuth } from "@/lib/auth-context";

export const Route = createFileRoute("/change-password")({ component: ChangePassword });

function ChangePassword() {
  const { changePassword, loading, nextStep } = useAuth();
  const navigate = useNavigate();
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    // Single source of truth for navigation: the submit handler only calls
    // the API, this effect reacts to the resulting nextStep. Previously the
    // handler also navigated straight to /chat after success, which raced
    // this effect (nextStep flips away from PASSWORD_CHANGE_REQUIRED at the
    // same time) and could bounce the user back to /signin instead.
    if (loading) return;
    const target = resolveAuthDestination(nextStep);
    if (target !== "/change-password") navigate({ to: target, replace: true });
  }, [loading, nextStep, navigate]);
  return <AuthShell><h1 className="text-2xl font-semibold">Set your permanent password</h1>
    <p className="mt-2 text-sm text-muted-foreground">Your temporary password has been consumed. Complete this step to activate the account.</p>
    <form className="mt-8 space-y-4" onSubmit={async (event) => {
      event.preventDefault(); setError("");
      if (password !== confirmation) { setError("Passwords do not match."); return; }
      try { await changePassword(password); }
      catch (value) { setError(value instanceof Error ? value.message : "Password change failed."); }
    }}>
      <div className="space-y-1.5"><Label>New password</Label><Input type="password" minLength={12} value={password} onChange={(e) => setPassword(e.target.value)} required /></div>
      <div className="space-y-1.5"><Label>Confirm password</Label><Input type="password" minLength={12} value={confirmation} onChange={(e) => setConfirmation(e.target.value)} required /></div>
      {error && <p className="text-sm text-destructive">{error}</p>}<Button type="submit" className="w-full">Activate account</Button>
    </form>
  </AuthShell>;
}

