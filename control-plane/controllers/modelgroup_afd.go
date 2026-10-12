// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

package controllers

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"time"

	inferencev1alpha1 "github.com/shiweijiezero/foretoken/control-plane/api/v1alpha1"
	vllmconfig "github.com/shiweijiezero/foretoken/control-plane/internal/vllm"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const (
	afdExecutionRoleLabel = "inference.foretoken.io/execution-role"
	afdMemberIndexLabel   = "inference.foretoken.io/member-index"
	afdMemberFinalizer    = "inference.foretoken.io/afd-member"
)

func afdMemberName(group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole) string {
	return "afd-" + string(group.UID) + "-" + string(role) + "-0"
}

func afdRendezvousName(group *inferencev1alpha1.ModelGroup) string {
	return "afd-" + string(group.UID) + "-f0"
}

// afdMemberLaunchPlan supplies the persisted Group identity and F0 DNS name to the V2 adapter.
func afdMemberLaunchPlan(group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole) (vllmconfig.LaunchPlan, error) {
	for _, side := range []inferencev1alpha1.AFDMemberSpec{group.Spec.AFD.Attention, group.Spec.AFD.FFN} {
		if side.MemberCount != 1 || side.Resources.Requests.GPU.Count != 1 {
			return vllmconfig.LaunchPlan{}, fmt.Errorf("AFD workload materialization currently requires one single-GPU member per role")
		}
	}
	return vllmconfig.BuildAFDLaunchPlan(group.Spec, vllmconfig.AFDMember{
		Role: role, Index: 0, GroupUID: string(group.UID),
		RendezvousHost: afdRendezvousName(group) + "." + group.Namespace + ".svc",
	})
}

// modelPreparationLaunchPlan uses the same artifact contract without starting either engine.
func modelPreparationLaunchPlan(group *inferencev1alpha1.ModelGroup) (vllmconfig.LaunchPlan, error) {
	if group.Spec.AFD != nil {
		return afdMemberLaunchPlan(group, inferencev1alpha1.AFDAttention)
	}
	return vllmconfig.BuildLaunchPlan(group.Spec)
}

// desiredAFDMember reuses the common runtime template with role resources and a non-restarting Pod.
func (reconciler *ModelGroupReconciler) desiredAFDMember(ctx context.Context, group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole) (*corev1.Pod, error) {
	plan, err := afdMemberLaunchPlan(group, role)
	if err != nil {
		return nil, err
	}
	encoded, err := plan.JSON()
	if err != nil {
		return nil, err
	}
	resources := &group.Spec.AFD.Attention.Resources
	if role == inferencev1alpha1.AFDFFN {
		resources = &group.Spec.AFD.FFN.Resources
	}
	template, err := desiredModelServerTemplate(group, resources, reconciler.ImagePullSecrets, reconciler.ApplicationFiles, encoded, "FORETOKEN_VLLM_LAUNCH_PLAN", "foretoken-model-server", plan.Lifecycle.StartupSeconds, plan.Lifecycle.DrainSeconds)
	if err != nil {
		return nil, err
	}
	template.Labels[afdExecutionRoleLabel], template.Labels[afdMemberIndexLabel] = string(role), "0"
	template.Spec.RestartPolicy = corev1.RestartPolicyNever
	// Never applies to probe-triggered exits too; no member may restart into this domain.
	template.Spec.Containers[0].Env = append(template.Spec.Containers[0].Env, corev1.EnvVar{
		Name: "FORETOKEN_MEMBER_IP", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}},
	})
	if role == inferencev1alpha1.AFDFFN {
		template.Spec.Containers[0].Ports = append(template.Spec.Containers[0].Ports, corev1.ContainerPort{Name: "afd", ContainerPort: group.Spec.AFD.Port, Protocol: corev1.ProtocolTCP})
	}
	if err := placeRuntimeCache(ctx, reconciler.Client, group.Namespace, group.Spec.Artifacts.Cache, template); err != nil {
		return nil, err
	}
	pod := &corev1.Pod{ObjectMeta: template.ObjectMeta, Spec: template.Spec}
	pod.Name, pod.Namespace = afdMemberName(group, role), group.Namespace
	// Retain the Pod even if deletion races the Group status write. The same owner records its
	// identity and real termination before releasing it; absence never authorizes a replacement.
	controllerutil.AddFinalizer(pod, afdMemberFinalizer)
	if err := controllerutil.SetControllerReference(group, pod, reconciler.Scheme()); err != nil {
		return nil, err
	}
	return pod, nil
}

// reconcileAFDRendezvous owns a headless, readiness-independent F0 endpoint for this Group UID.
func (reconciler *ModelGroupReconciler) reconcileAFDRendezvous(ctx context.Context, group *inferencev1alpha1.ModelGroup) error {
	labels := modelGroupLabels(group)
	labels[afdExecutionRoleLabel], labels[afdMemberIndexLabel] = string(inferencev1alpha1.AFDFFN), "0"
	desired := &corev1.Service{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Service"},
		ObjectMeta: metav1.ObjectMeta{Name: afdRendezvousName(group), Namespace: group.Namespace},
		Spec: corev1.ServiceSpec{ClusterIP: corev1.ClusterIPNone, PublishNotReadyAddresses: true, Selector: labels,
			Ports: []corev1.ServicePort{{Name: "afd", Port: group.Spec.AFD.Port, TargetPort: intstr.FromString("afd"), Protocol: corev1.ProtocolTCP}},
		},
	}
	if err := controllerutil.SetControllerReference(group, desired, reconciler.Scheme()); err != nil {
		return err
	}
	current := new(corev1.Service)
	err := reconciler.APIReader.Get(ctx, client.ObjectKeyFromObject(desired), current)
	if err == nil && !metav1.IsControlledBy(current, group) {
		return fmt.Errorf("AFD rendezvous %q belongs to another owner", current.Name)
	}
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return reconciler.Patch(ctx, desired, client.Apply, client.FieldOwner(modelGroupFieldOwner), client.ForceOwnership)
}

// observeAFDMember reads the API server directly so informer lag cannot become a lost member.
// The persisted UID and failure are retained even when a Pod has disappeared.
func (reconciler *ModelGroupReconciler) observeAFDMember(ctx context.Context, group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole) (*corev1.Pod, *inferencev1alpha1.ModelGroupAFDMemberStatus, error) {
	var member *inferencev1alpha1.ModelGroupAFDMemberStatus
	for i := range group.Status.AFDMembers {
		if group.Status.AFDMembers[i].Role == role {
			member = &group.Status.AFDMembers[i]
			break
		}
	}
	pod := new(corev1.Pod)
	err := reconciler.APIReader.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: afdMemberName(group, role)}, pod)
	if apierrors.IsNotFound(err) {
		if member != nil && group.DeletionTimestamp.IsZero() && member.FailureReason == "" {
			member.FailureReason, member.FailureMessage = "MemberMissing", fmt.Sprintf("%s/0 Pod %s disappeared; termination is not inferred", role, member.PodUID)
		}
		return nil, member, nil
	}
	if err != nil {
		return nil, member, err
	}
	if !metav1.IsControlledBy(pod, group) || pod.Labels[modelGroupUIDLabel] != string(group.UID) || pod.Labels[afdExecutionRoleLabel] != string(role) || pod.Labels[afdMemberIndexLabel] != "0" {
		return nil, member, fmt.Errorf("AFD Pod %q has an unexpected owner or member identity", pod.Name)
	}
	if member == nil {
		group.Status.AFDMembers = append(group.Status.AFDMembers, inferencev1alpha1.ModelGroupAFDMemberStatus{Role: role, PodUID: string(pod.UID)})
		member = &group.Status.AFDMembers[len(group.Status.AFDMembers)-1]
	}
	if member.PodUID != string(pod.UID) {
		if !group.DeletionTimestamp.IsZero() {
			return nil, member, fmt.Errorf("AFD Pod %q has a replacement UID; cleanup requires owner intervention", pod.Name)
		}
		member.FailureReason, member.FailureMessage = "MemberReplaced", fmt.Sprintf("%s/0 no longer has recorded Pod UID %s; replacement is not adopted", role, member.PodUID)
		return nil, member, nil
	}
	if afdPodStopped(pod) {
		member.TerminationConfirmed = true
	}
	if group.DeletionTimestamp.IsZero() && member.FailureReason == "" {
		member.FailureReason, member.FailureMessage = afdMemberFailure(pod, group.Spec.Timeouts.Startup, reconciler.now())
	}
	return pod, member, nil
}

// afdPodStopped requires terminal phase/container evidence for a scheduled Pod, not deletion or eviction alone.
func afdPodStopped(pod *corev1.Pod) bool {
	if pod.Status.Reason == "NodeLost" || pod.Status.Phase == corev1.PodUnknown {
		return false
	}
	if pod.Spec.NodeName == "" {
		return !pod.DeletionTimestamp.IsZero()
	}
	// Kubelet terminalizes a gracefully deleted Pod even when image pull or init never completed.
	// Node-controller loss and eviction alone are not that evidence.
	if pod.Status.Phase == corev1.PodSucceeded || (pod.Status.Phase == corev1.PodFailed && pod.Status.Reason != "Evicted") {
		unknown := false
		for _, status := range append(pod.Status.ContainerStatuses, pod.Status.InitContainerStatuses...) {
			if status.State.Running != nil || (status.State.Terminated != nil && status.State.Terminated.Reason == "ContainerStatusUnknown") {
				unknown = true
			}
		}
		if !unknown {
			return true
		}
	}
	for _, container := range pod.Spec.Containers {
		stopped := false
		for _, status := range pod.Status.ContainerStatuses {
			if status.Name == container.Name && status.State.Terminated != nil && status.State.Terminated.Reason != "ContainerStatusUnknown" {
				stopped = true
				break
			}
		}
		if !stopped {
			return false
		}
	}
	for _, status := range pod.Status.InitContainerStatuses {
		if status.State.Running != nil {
			return false
		}
	}
	return true
}

// afdMemberFailure classifies C08 inputs without equating failure with released resources.
func afdMemberFailure(pod *corev1.Pod, startup inferencev1alpha1.Duration, now time.Time) (string, string) {
	detail := fmt.Sprintf("%s: phase=%s reason=%s", pod.Name, pod.Status.Phase, pod.Status.Reason)
	if pod.Status.Reason == "NodeLost" || pod.Status.Phase == corev1.PodUnknown {
		return "NodeUnavailable", detail + "; termination unconfirmed"
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && (condition.Status == corev1.ConditionUnknown || condition.Reason == "NodeNotReady" || condition.Reason == "NodeStatusUnknown") {
			return "NodeUnavailable", detail + "; " + condition.Reason + "; termination unconfirmed"
		}
	}
	if pod.Status.Phase == corev1.PodFailed {
		return "MemberFailed", detail
	}
	if pod.Status.Phase == corev1.PodSucceeded {
		return "MemberSucceeded", detail
	}
	if !pod.DeletionTimestamp.IsZero() {
		return "MemberDeleting", detail
	}
	for _, status := range pod.Status.ContainerStatuses {
		if status.State.Terminated != nil {
			return "ContainerTerminated", fmt.Sprintf("%s container %s: reason=%s exitCode=%d", pod.Name, status.Name, status.State.Terminated.Reason, status.State.Terminated.ExitCode)
		}
		if status.Name == "model-server" && status.State.Running != nil && (status.Started == nil || !*status.Started) {
			timeout, _ := time.ParseDuration(string(startup))
			if now.After(status.State.Running.StartedAt.Add(timeout)) && !podReady(pod) {
				return "StartupTimedOut", detail
			}
		}
	}
	return "", ""
}

// saveAFDMemberStatus persists identity/termination before any further creation or finalizer removal.
func (reconciler *ModelGroupReconciler) saveAFDMemberStatus(ctx context.Context, group, base *inferencev1alpha1.ModelGroup) error {
	if reflect.DeepEqual(base.Status, group.Status) {
		return nil
	}
	return reconciler.Status().Patch(ctx, group, client.MergeFrom(base))
}

// reconcileAFD creates only never-created members and publishes the complete pair's serving state.
// Latched member failure withdraws admission without replacing any member in the old domain.
func (reconciler *ModelGroupReconciler) reconcileAFD(ctx context.Context, group *inferencev1alpha1.ModelGroup, pool *inferencev1alpha1.ModelPool) (ctrl.Result, error) {
	// Member and Group observations must share the API-server consistency boundary.
	if err := reconciler.APIReader.Get(ctx, client.ObjectKeyFromObject(group), group); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !group.DeletionTimestamp.IsZero() {
		return reconciler.reconcileAFDDelete(ctx, group)
	}
	base := group.DeepCopy()
	roles := []inferencev1alpha1.AFDExecutionRole{inferencev1alpha1.AFDAttention, inferencev1alpha1.AFDFFN}
	for _, role := range roles {
		if _, _, err := reconciler.observeAFDMember(ctx, group, role); err != nil {
			state := modelGroupMaterializedState(false, schedulingNotEvaluated())
			state.unavailable = &modelGroupConditionState{status: metav1.ConditionFalse, reason: "MemberObservationFailed", message: fmt.Sprintf("%s/0: %v", role, err)}
			return ctrl.Result{}, errors.Join(err, reconciler.publishAFDState(ctx, group, pool, state))
		}
	}
	memberStatusErr := reconciler.saveAFDMemberStatus(ctx, group, base)
	for _, member := range group.Status.AFDMembers {
		if member.FailureReason != "" {
			state := modelGroupMaterializedState(false, schedulingNotEvaluated())
			state.phase = inferencev1alpha1.ModelGroupPhaseFailed
			state.unavailable = &modelGroupConditionState{status: metav1.ConditionFalse, reason: member.FailureReason, message: fmt.Sprintf("%s/0: %s", member.Role, member.FailureMessage)}
			if len(group.Status.AFDMembers) != int(group.Spec.MemberCount) {
				state.materialized = modelGroupConditionState{status: metav1.ConditionFalse, reason: "IncompleteMembers", message: "Member failure stopped partial materialization"}
			}
			publicationErr := reconciler.publishAFDState(ctx, group, pool, state)
			return ctrl.Result{RequeueAfter: 10 * time.Second}, errors.Join(memberStatusErr, publicationErr)
		}
	}
	if memberStatusErr != nil {
		return ctrl.Result{}, memberStatusErr
	}
	if err := reconciler.reconcileAFDRendezvous(ctx, group); err != nil {
		return ctrl.Result{}, err
	}
	for _, role := range roles {
		exists := false
		for _, member := range group.Status.AFDMembers {
			if member.Role == role {
				exists = true
				break
			}
		}
		if exists {
			continue
		}
		desired, err := reconciler.desiredAFDMember(ctx, group, role)
		if err != nil {
			return ctrl.Result{}, err
		}
		if err := reconciler.Create(ctx, desired); err != nil {
			return ctrl.Result{}, err
		}
		base = group.DeepCopy()
		group.Status.AFDMembers = append(group.Status.AFDMembers, inferencev1alpha1.ModelGroupAFDMemberStatus{Role: role, PodUID: string(desired.UID)})
		if err := reconciler.saveAFDMemberStatus(ctx, group, base); err != nil {
			return ctrl.Result{}, err
		}
	}
	condition, observationErr := reconciler.afdServingCondition(ctx, group)
	if observationErr != nil {
		condition = modelGroupConditionState{status: metav1.ConditionFalse, reason: "MemberObservationFailed", message: observationErr.Error()}
	}
	scheduling, schedulingErr := reconciler.schedulingCapacity(ctx, group)
	if schedulingErr != nil {
		scheduling = schedulingNotEvaluated()
		condition = modelGroupConditionState{status: metav1.ConditionFalse, reason: "SchedulingObservationFailed", message: schedulingErr.Error()}
	}
	if condition.status == metav1.ConditionTrue {
		if err := reconciler.setAFDServing(ctx, group, true); err != nil {
			condition = modelGroupConditionState{status: metav1.ConditionFalse, reason: "AttentionAdmissionUnavailable", message: err.Error()}
		}
	}
	state := modelGroupMaterializedState(condition.status == metav1.ConditionTrue, scheduling)
	state.unavailable = &condition
	return ctrl.Result{RequeueAfter: 5 * time.Second}, errors.Join(observationErr, schedulingErr, reconciler.publishAFDState(ctx, group, pool, state))
}

// reconcileAFDDelete stops Attention before FFN, retaining evidence and discovery until both have ended.
// It never recreates members and never treats a missing running Pod as proof of GPU release.
func (reconciler *ModelGroupReconciler) reconcileAFDDelete(ctx context.Context, group *inferencev1alpha1.ModelGroup) (result ctrl.Result, reconcileErr error) {
	if err := reconciler.APIReader.Get(ctx, client.ObjectKeyFromObject(group), group); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if group.Status.DrainStartedAt == nil {
		now := metav1.NewTime(reconciler.now())
		if err := reconciler.updateDrainStatus(ctx, group, &now, false, false, false, "StoppingMembers", "Stopping Attention before FFN"); err != nil {
			return ctrl.Result{}, err
		}
	}
	// Revoke generation discovery before the existing ordered member shutdown.
	// Runtime SIGTERM closes admission and drains Attention before FFN is stopped.
	admissionErr := reconciler.setAFDServing(ctx, group, false)
	// HTTP loss cannot prevent the existing Pod shutdown path or prove termination.
	defer func() { reconcileErr = errors.Join(reconcileErr, admissionErr) }()
	if err := reconciler.withdrawAFDService(ctx, group); err != nil {
		return ctrl.Result{}, err
	}
	for _, role := range []inferencev1alpha1.AFDExecutionRole{inferencev1alpha1.AFDAttention, inferencev1alpha1.AFDFFN} {
		base := group.DeepCopy()
		pod, member, err := reconciler.observeAFDMember(ctx, group, role)
		if err != nil {
			return ctrl.Result{}, err
		}
		if err := reconciler.saveAFDMemberStatus(ctx, group, base); err != nil {
			return ctrl.Result{}, err
		}
		if pod == nil {
			if member != nil && !member.TerminationConfirmed {
				return ctrl.Result{RequeueAfter: modelGroupDrainPollInterval}, reconciler.updateDrainStatus(ctx, group, group.Status.DrainStartedAt, false, false, false, "TerminationUnconfirmed", fmt.Sprintf("%s/0 disappeared without termination evidence", role))
			}
			continue
		}
		if pod.DeletionTimestamp.IsZero() {
			if err := reconciler.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}); client.IgnoreNotFound(err) != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{RequeueAfter: modelGroupDrainPollInterval}, nil
		}
		if !member.TerminationConfirmed {
			return ctrl.Result{RequeueAfter: modelGroupDrainPollInterval}, reconciler.updateDrainStatus(ctx, group, group.Status.DrainStartedAt, false, false, false, "TerminationUnconfirmed", fmt.Sprintf("Waiting for %s/0 termination evidence", role))
		}
		if controllerutil.ContainsFinalizer(pod, afdMemberFinalizer) {
			before := pod.DeepCopy()
			controllerutil.RemoveFinalizer(pod, afdMemberFinalizer)
			if err := reconciler.Patch(ctx, pod, client.MergeFrom(before)); err != nil {
				return ctrl.Result{}, err
			}
		}
		return ctrl.Result{RequeueAfter: modelGroupDrainPollInterval}, nil
	}
	service := new(corev1.Service)
	err := reconciler.APIReader.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: afdRendezvousName(group)}, service)
	if err == nil {
		if !metav1.IsControlledBy(service, group) {
			return ctrl.Result{}, fmt.Errorf("AFD rendezvous belongs to another owner")
		}
		if err := reconciler.Delete(ctx, service, client.Preconditions{UID: &service.UID}); client.IgnoreNotFound(err) != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: modelGroupDrainPollInterval}, nil
	}
	if !apierrors.IsNotFound(err) {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, reconciler.removeDrainFinalizer(ctx, group)
}
