// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

package controllers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	inferencev1alpha1 "github.com/shiweijiezero/foretoken/control-plane/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// afdRuntimeStatus is produced only after native startup and the actual connector/loop events.
// The model-server's existing health owner revokes Ready on runtime failure.
type afdRuntimeStatus struct {
	GroupUID    string                             `json:"groupUid"`
	Role        inferencev1alpha1.AFDExecutionRole `json:"role"`
	MemberIndex *int32                             `json:"memberIndex"`
	MemberCount int32                              `json:"memberCount"`
	Ready       bool                               `json:"ready"`
}

// afdServingCondition observes the exact persisted pair and both private runtime endpoints.
// Pod readiness alone is insufficient; no listener during startup means unavailable.
func (reconciler *ModelGroupReconciler) afdServingCondition(ctx context.Context, group *inferencev1alpha1.ModelGroup) (modelGroupConditionState, error) {
	condition := modelGroupConditionState{status: metav1.ConditionFalse, reason: "MembersNotReady"}
	var pods corev1.PodList
	if err := reconciler.APIReader.List(ctx, &pods, client.InNamespace(group.Namespace), client.MatchingLabels{modelGroupUIDLabel: string(group.UID)}); err != nil {
		return condition, err
	}
	var blockers []string
	if len(pods.Items) != int(group.Spec.MemberCount) {
		condition.reason = "IncompleteMembers"
		blockers = append(blockers, fmt.Sprintf("Expected %d AFD members, observed %d", group.Spec.MemberCount, len(pods.Items)))
	}
	for _, role := range []inferencev1alpha1.AFDExecutionRole{inferencev1alpha1.AFDAttention, inferencev1alpha1.AFDFFN} {
		var pod *corev1.Pod
		for i := range pods.Items {
			if pods.Items[i].Name == afdMemberName(group, role) {
				pod = &pods.Items[i]
			}
		}
		identityMatches := false
		if pod != nil && metav1.IsControlledBy(pod, group) && pod.Labels[afdExecutionRoleLabel] == string(role) && pod.Labels[afdMemberIndexLabel] == "0" {
			for _, member := range group.Status.AFDMembers {
				if member.Role == role && member.PodUID == string(pod.UID) && member.FailureReason == "" {
					identityMatches = true
				}
			}
		}
		if !identityMatches {
			blockers = append(blockers, fmt.Sprintf("%s/0: expected persisted member identity is absent", role))
			continue
		}
		if pod.Spec.NodeName == "" {
			blockers = append(blockers, fmt.Sprintf("%s/0: waiting for scheduling", role))
			continue
		}
		if !pod.DeletionTimestamp.IsZero() || pod.Status.Phase != corev1.PodRunning || !podReady(pod) || pod.Status.PodIP == "" {
			blockers = append(blockers, fmt.Sprintf("%s/0: phase=%s, Pod is not ready", role, pod.Status.Phase))
			continue
		}
		endpoint := afdPodEndpoint(group, pod)
		var status afdRuntimeStatus
		if err := readAFDStatus(ctx, endpoint+"/internal/afd/status", &status); err != nil {
			blockers = append(blockers, fmt.Sprintf("%s/0: runtime status unavailable: %v", role, err))
			continue
		}
		if status.GroupUID != string(group.UID) || status.Role != role || status.MemberIndex == nil || *status.MemberIndex != 0 || status.MemberCount != 1 || !status.Ready {
			blockers = append(blockers, fmt.Sprintf("%s/0: runtime identity or complete initialization is not ready", role))
			continue
		}
	}
	if len(blockers) > 0 {
		condition.message = strings.Join(blockers, "; ")
		return condition, nil
	}
	return modelGroupConditionState{status: metav1.ConditionTrue, reason: "AFDReady", message: "Attention and FFN identities, weights, engines, communication, FFN loop and Attention admission are ready"}, nil
}

// readAFDStatus consumes the existing runtime JSON contracts with a bounded observation request.
func readAFDStatus(ctx context.Context, endpoint string, status any) error {
	return requestAFDStatus(ctx, http.MethodGet, endpoint, nil, status)
}

// requestAFDStatus shares bounded JSON transport for observation and Attention admission.
func requestAFDStatus(ctx context.Context, method, endpoint string, body []byte, status any) error {
	request, err := http.NewRequestWithContext(ctx, method, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := (&http.Client{Timeout: 2 * time.Second}).Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", response.StatusCode)
	}
	return json.NewDecoder(response.Body).Decode(status)
}

func afdPodEndpoint(group *inferencev1alpha1.ModelGroup, pod *corev1.Pod) string {
	return "http://" + net.JoinHostPort(pod.Status.PodIP, strconv.Itoa(int(group.Spec.Runtime.Port)))
}

// publishAFDState installs the Attention Service before Ready, or publishes unavailability.
// Reconcile's withdrawal stage also runs when publication or preparation fails.
func (reconciler *ModelGroupReconciler) publishAFDState(ctx context.Context, group *inferencev1alpha1.ModelGroup, pool *inferencev1alpha1.ModelPool, state modelGroupStatusState) error {
	if state.available {
		if err := reconciler.reconcileService(ctx, group, pool); err != nil {
			state.available = false
			state.phase = inferencev1alpha1.ModelGroupPhaseProvisioning
			state.unavailable = &modelGroupConditionState{status: metav1.ConditionFalse, reason: "ServingPublicationFailed", message: err.Error()}
			return errors.Join(err, reconciler.updateStatus(ctx, group, state))
		}
	}
	return reconciler.updateStatus(ctx, group, state)
}

// withdrawAFDServing is Reconcile's unavailability exit, including prerequisite API errors.
// It preserves preparation/member facts and attempts each serving withdrawal despite errors.
func (reconciler *ModelGroupReconciler) withdrawAFDServing(ctx context.Context, group *inferencev1alpha1.ModelGroup, cause error) error {
	var statusErr error
	if cause != nil {
		base := group.DeepCopy()
		group.Status.ReadyMembers = 0
		if group.Status.Phase == inferencev1alpha1.ModelGroupPhaseReady {
			group.Status.Phase = inferencev1alpha1.ModelGroupPhaseProvisioning
		}
		condition := metav1.Condition{Status: metav1.ConditionFalse, Reason: "ServingReconcileFailed", Message: cause.Error(), ObservedGeneration: group.Generation}
		if previous := meta.FindStatusCondition(group.Status.Conditions, conditionReady); previous != nil && previous.Status == metav1.ConditionFalse && group.Status.Phase == inferencev1alpha1.ModelGroupPhaseFailed {
			condition.Reason, condition.Message = previous.Reason, previous.Message
		}
		for _, kind := range []string{conditionReady, conditionWorkloadAvailable} {
			condition.Type = kind
			meta.SetStatusCondition(&group.Status.Conditions, condition)
		}
		statusErr = reconciler.Status().Patch(ctx, group, client.MergeFrom(base))
	}
	serviceErr := reconciler.withdrawAFDService(ctx, group)
	admissionErr := reconciler.setAFDServing(ctx, group, false)
	return errors.Join(cause, statusErr, serviceErr, admissionErr)
}

// withdrawAFDService removes generation discovery, retaining the independent F0 rendezvous.
// Deletion is repeatable after controller restart and never changes member or discovery ownership.
func (reconciler *ModelGroupReconciler) withdrawAFDService(ctx context.Context, group *inferencev1alpha1.ModelGroup) error {
	service := new(corev1.Service)
	err := reconciler.APIReader.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: modelGroupServiceName(group)}, service)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if !metav1.IsControlledBy(service, group) {
		return fmt.Errorf("AFD generation Service %q has another owner", service.Name)
	}
	return client.IgnoreNotFound(reconciler.Delete(ctx, service, client.Preconditions{UID: &service.UID}))
}

// setAFDServing controls the persisted Attention member's reversible serving gate.
// It cannot reopen lifecycle admission closed for drain or shutdown.
func (reconciler *ModelGroupReconciler) setAFDServing(ctx context.Context, group *inferencev1alpha1.ModelGroup, enabled bool) error {
	pod := new(corev1.Pod)
	err := reconciler.APIReader.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: afdMemberName(group, inferencev1alpha1.AFDAttention)}, pod)
	if apierrors.IsNotFound(err) && !enabled {
		return nil
	}
	if err != nil {
		return err
	}
	for _, member := range group.Status.AFDMembers {
		if member.Role == inferencev1alpha1.AFDAttention && member.PodUID == string(pod.UID) && metav1.IsControlledBy(pod, group) && pod.Labels[modelGroupUIDLabel] == string(group.UID) && pod.Labels[afdExecutionRoleLabel] == string(member.Role) && pod.Labels[afdMemberIndexLabel] == "0" {
			if !enabled && pod.Status.PodIP == "" {
				return nil
			}
			request := struct {
				GroupUID    string `json:"groupUid"`
				MemberIndex int32  `json:"memberIndex"`
				Enabled     bool   `json:"enabled"`
			}{string(group.UID), 0, enabled}
			body, err := json.Marshal(request)
			if err != nil {
				return err
			}
			var response struct {
				afdRuntimeStatus
				Accepting *bool `json:"accepting"`
			}
			if err := requestAFDStatus(ctx, http.MethodPost, afdPodEndpoint(group, pod)+"/internal/afd/serving", body, &response); err != nil {
				return fmt.Errorf("attention/0 serving gate: %w", err)
			}
			if response.GroupUID != string(group.UID) || response.Role != member.Role || response.MemberIndex == nil || *response.MemberIndex != 0 || response.MemberCount != 1 || response.Accepting == nil {
				return fmt.Errorf("attention/0 serving gate returned an invalid identity or admission state")
			}
			if *response.Accepting != enabled || (enabled && !response.Ready) {
				return fmt.Errorf("attention/0 did not confirm requested serving admission")
			}
			return nil
		}
	}
	return fmt.Errorf("cannot control admission on an unrecognized Attention member")
}
