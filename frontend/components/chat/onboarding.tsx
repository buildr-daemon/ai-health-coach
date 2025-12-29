"use client";

import React, { useState, FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Heart, User, Calendar, Users } from "lucide-react";
import { cn } from "@/lib/utils";

// === Types ===

interface OnboardingFormData {
  displayName: string;
  ageYears: number | "";
  biologicalSex: "male" | "female" | "other" | "";
}

interface OnboardingProps {
  userId: number;
  onOnboardingComplete: () => void;
  apiBaseUrl: string;
}

// === Onboarding Component ===

export function Onboarding({
  userId,
  onOnboardingComplete,
  apiBaseUrl,
}: OnboardingProps) {
  const [formData, setFormData] = useState<OnboardingFormData>({
    displayName: "",
    ageYears: "",
    biologicalSex: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { id: "name", title: "What should we call you?", icon: User },
    { id: "age", title: "How old are you?", icon: Calendar },
    { id: "sex", title: "Biological sex", icon: Users },
  ];

  const canProceed = () => {
    switch (currentStep) {
      case 0:
        return formData.displayName.trim().length >= 2;
      case 1:
        return (
          formData.ageYears !== "" &&
          formData.ageYears >= 1 &&
          formData.ageYears <= 120
        );
      case 2:
        return formData.biologicalSex !== "";
      default:
        return false;
    }
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!canProceed()) return;

    if (currentStep < steps.length - 1) {
      handleNext();
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/chat/onboarding`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          display_name: formData.displayName.trim(),
          age_years: formData.ageYears,
          biological_sex: formData.biologicalSex,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail?.error_message || "Failed to complete onboarding"
        );
      }

      onOnboardingComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setIsSubmitting(false);
    }
  };

  const CurrentStepIcon = steps[currentStep].icon;

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-whatsapp-dark-bg px-4 py-8">
      {/* Header */}
      <div className="flex flex-col items-center mb-8 animate-in fade-in slide-in-from-top-4 duration-500">
        <div className="size-20 rounded-full bg-gradient-to-br from-whatsapp-teal to-ios-green flex items-center justify-center mb-4 shadow-lg shadow-whatsapp-teal/20">
          <Heart className="size-10 text-white" />
        </div>
        <h1 className="text-2xl font-bold text-white mb-2">
          Welcome to Health Agent
        </h1>
        <p className="text-gray-400 text-center max-w-sm">
          Let&apos;s get to know you so we can provide personalized health
          guidance
        </p>
      </div>

      {/* Progress Indicator */}
      <div className="flex items-center gap-2 mb-8">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={cn(
              "size-2.5 rounded-full transition-all duration-300",
              index === currentStep
                ? "bg-whatsapp-teal w-8"
                : index < currentStep
                ? "bg-whatsapp-teal"
                : "bg-gray-600"
            )}
          />
        ))}
      </div>

      {/* Form Card */}
      <div className="w-full max-w-md bg-whatsapp-header-bg rounded-2xl p-6 shadow-xl animate-in fade-in slide-in-from-bottom-4 duration-500">
        <form onSubmit={handleSubmit}>
          {/* Step Header */}
          <div className="flex items-center gap-3 mb-6">
            <div className="size-10 rounded-full bg-whatsapp-teal/20 flex items-center justify-center">
              <CurrentStepIcon className="size-5 text-whatsapp-teal" />
            </div>
            <h2 className="text-lg font-semibold text-white">
              {steps[currentStep].title}
            </h2>
          </div>

          {/* Step Content */}
          <div className="min-h-[120px]">
            {currentStep === 0 && (
              <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                <div className="space-y-2">
                  <Label htmlFor="displayName" className="text-gray-300">
                    Your name
                  </Label>
                  <Input
                    id="displayName"
                    type="text"
                    placeholder="Enter your name"
                    value={formData.displayName}
                    onChange={(e) =>
                      setFormData({ ...formData, displayName: e.target.value })
                    }
                    className={cn(
                      "h-12 bg-whatsapp-input-bg border-whatsapp-border",
                      "text-white placeholder:text-gray-500",
                      "focus-visible:ring-whatsapp-teal focus-visible:border-whatsapp-teal"
                    )}
                    maxLength={100}
                    autoFocus
                  />
                </div>
                <p className="text-xs text-gray-500">
                  This is how we&apos;ll address you in our conversations
                </p>
              </div>
            )}

            {currentStep === 1 && (
              <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                <div className="space-y-2">
                  <Label htmlFor="ageYears" className="text-gray-300">
                    Age (years)
                  </Label>
                  <Input
                    id="ageYears"
                    type="number"
                    placeholder="Enter your age"
                    value={formData.ageYears}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        ageYears:
                          e.target.value === ""
                            ? ""
                            : parseInt(e.target.value, 10),
                      })
                    }
                    className={cn(
                      "h-12 bg-whatsapp-input-bg border-whatsapp-border",
                      "text-white placeholder:text-gray-500",
                      "focus-visible:ring-whatsapp-teal focus-visible:border-whatsapp-teal"
                    )}
                    min={1}
                    max={120}
                    autoFocus
                  />
                </div>
                <p className="text-xs text-gray-500">
                  Your age helps us provide age-appropriate health guidance
                </p>
              </div>
            )}

            {currentStep === 2 && (
              <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                <div className="space-y-2">
                  <Label className="text-gray-300">Biological sex</Label>
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { value: "male", label: "Male" },
                      { value: "female", label: "Female" },
                      { value: "other", label: "Other" },
                    ].map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() =>
                          setFormData({
                            ...formData,
                            biologicalSex: option.value as
                              | "male"
                              | "female"
                              | "other",
                          })
                        }
                        className={cn(
                          "h-12 rounded-lg border transition-all duration-200",
                          "text-sm font-medium",
                          formData.biologicalSex === option.value
                            ? "bg-whatsapp-teal border-whatsapp-teal text-white"
                            : "bg-whatsapp-input-bg border-whatsapp-border text-gray-300 hover:border-whatsapp-teal/50"
                        )}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
                <p className="text-xs text-gray-500">
                  This helps us tailor health advice to your biology
                </p>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 animate-in fade-in duration-200">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {/* Navigation */}
          <div className="flex items-center gap-3 mt-6">
            {currentStep > 0 && (
              <Button
                type="button"
                variant="outline"
                onClick={handleBack}
                disabled={isSubmitting}
                className="flex-1 h-12 bg-transparent border-whatsapp-border text-gray-300 hover:bg-whatsapp-input-bg hover:text-white"
              >
                Back
              </Button>
            )}
            <Button
              type="submit"
              disabled={!canProceed() || isSubmitting}
              className={cn(
                "flex-1 h-12",
                "bg-whatsapp-teal hover:bg-whatsapp-teal/80",
                "disabled:bg-gray-600 disabled:opacity-50"
              )}
            >
              {isSubmitting ? (
                <Spinner className="size-5 text-white" />
              ) : currentStep === steps.length - 1 ? (
                "Get Started"
              ) : (
                "Continue"
              )}
            </Button>
          </div>
        </form>
      </div>

      {/* Footer */}
      <p className="text-xs text-gray-500 mt-6 text-center max-w-sm">
        Your information is stored securely and used only to provide
        personalized health guidance
      </p>
    </div>
  );
}

export default Onboarding;

