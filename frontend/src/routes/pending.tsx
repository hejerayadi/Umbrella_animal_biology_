import { Link, createFileRoute } from "@tanstack/react-router";
import { Clock3, MailCheck } from "lucide-react";

import { Button } from "@/components/ui/button";
import { AuthShell } from "@/components/umbrella/auth-shell";

export const Route = createFileRoute("/pending")({ component: Pending });

function Pending() {
  return <AuthShell><div className="text-center">
    <MailCheck className="mx-auto h-10 w-10 text-primary" />
    <h1 className="mt-5 text-2xl font-semibold">Check your email</h1>
    <p className="mt-2 text-sm text-muted-foreground">Verify your address within 30 minutes. An administrator will review your professional profile after verification.</p>
    <div className="mt-6 flex items-center justify-center gap-2 text-sm text-muted-foreground"><Clock3 className="h-4 w-4" /> Approval is required before sign in</div>
    <Button asChild variant="outline" className="mt-8"><Link to="/signin">Back to sign in</Link></Button>
  </div></AuthShell>;
}

