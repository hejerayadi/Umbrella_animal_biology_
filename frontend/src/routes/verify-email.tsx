import { Link, createFileRoute } from "@tanstack/react-router";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { AuthShell } from "@/components/umbrella/auth-shell";
import { apiRequest } from "@/lib/api-client";

export const Route = createFileRoute("/verify-email")({
  validateSearch: (search: Record<string, unknown>) => ({ token: typeof search.token === "string" ? search.token : "" }),
  component: VerifyEmail,
});

function VerifyEmail() {
  const { token } = Route.useSearch();
  const [state, setState] = useState<"loading" | "success" | "error">("loading");
  const [message, setMessage] = useState("");
  useEffect(() => {
    if (!token) { setState("error"); setMessage("The verification token is missing."); return; }
    apiRequest("/api/v1/auth/email-verification/confirm", { method: "POST", body: JSON.stringify({ token }) })
      .then(() => setState("success"))
      .catch((error: unknown) => { setState("error"); setMessage(error instanceof Error ? error.message : "Verification failed."); });
  }, [token]);
  return <AuthShell><div className="text-center">
    {state === "loading" && <Loader2 className="mx-auto h-9 w-9 animate-spin text-primary" />}
    {state === "success" && <CheckCircle2 className="mx-auto h-10 w-10 text-primary" />}
    {state === "error" && <XCircle className="mx-auto h-10 w-10 text-destructive" />}
    <h1 className="mt-5 text-2xl font-semibold">{state === "loading" ? "Verifying email" : state === "success" ? "Email verified" : "Verification failed"}</h1>
    <p className="mt-2 text-sm text-muted-foreground">{state === "success" ? "Your application is now waiting for administrator approval." : message}</p>
    {state !== "loading" && <Button asChild className="mt-8"><Link to="/signin">Continue</Link></Button>}
  </div></AuthShell>;
}

