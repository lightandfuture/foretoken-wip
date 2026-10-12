// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

package controllers

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"time"

	inferencev1alpha1 "github.com/shiweijiezero/foretoken/control-plane/api/v1alpha1"
	"github.com/shiweijiezero/foretoken/control-plane/internal/runtimeconfig"
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

// afdMemberIdentity is one expected role-local position in the immutable Group topology.
type afdMemberIdentity struct {
	role  inferencev1alpha1.AFDExecutionRole
	index int32
	count int32
}

// afdMembers enumerates the complete domain in Attention-then-FFN shutdown order.
func afdMembers(group *inferencev1alpha1.ModelGroup) []afdMemberIdentity {
	var members []afdMemberIdentity
	for _, role := range []inferencev1alpha1.AFDExecutionRole{inferencev1alpha1.AFDAttention, inferencev1alpha1.AFDFFN} {
		count := afdRoleSpec(group, role).MemberCount
		for index := int32(0); index < count; index++ {
			members = append(members, afdMemberIdentity{role, index, count})
		}
	}
	return members
}

func afdRoleSpec(group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole) inferencev1alpha1.AFDMemberSpec {
	if role == inferencev1alpha1.AFDAttention {
		return group.Spec.AFD.Attention
	}
	return group.Spec.AFD.FFN
}

func afdMemberName(group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole, index int32) string {
	return "afd-" + string(group.UID) + "-" + string(role) + "-" + strconv.Itoa(int(index))
}

func afdDiscoveryName(group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole) string {
	suffix := "f0"
	if role == inferencev1alpha1.AFDAttention {
		suffix = "a0"
	}
	return "afd-" + string(group.UID) + "-" + suffix
}

// afdRoleCoordination assigns private ports at materialization, separate from shared F0.
// All members derive the same endpoint without writing addresses into immutable spec.
func afdRoleCoordination(group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole) *vllmconfig.AFDCoordination {
	if afdRoleSpec(group, role).MemberCount == 1 {
		return nil
	}
	ports := make([]int32, 0, 2)
	for port := int32(29800); len(ports) < 2; port++ {
		if port == group.Spec.Runtime.Port || port == group.Spec.AFD.Port ||
			(group.Spec.Artifacts.Cache != nil && port == runtimeconfig.CacheObservationPort(group.Spec.Runtime.Port)) {
			continue
		}
		ports = append(ports, port)
	}
	return &vllmconfig.AFDCoordination{Host: afdDiscoveryName(group, role) + "." + group.Namespace + ".svc", MasterPort: ports[0], HandshakePort: ports[1]}
}

// afdMemberLaunchPlan supplies persisted identity and materialized endpoints to the V2 adapter.
func afdMemberLaunchPlan(group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole, index int32) (vllmconfig.LaunchPlan, error) {
	return vllmconfig.BuildAFDLaunchPlan(group.Spec, vllmconfig.AFDMember{
		Role: role, Index: index, GroupUID: string(group.UID),
		RendezvousHost: afdDiscoveryName(group, inferencev1alpha1.AFDFFN) + "." + group.Namespace + ".svc",
		Coordination:   afdRoleCoordination(group, role),
	})
}

// modelPreparationLaunchPlan uses the same artifact contract without starting either engine.
func modelPreparationLaunchPlan(group *inferencev1alpha1.ModelGroup) (vllmconfig.LaunchPlan, error) {
	if group.Spec.AFD != nil {
		return afdMemberLaunchPlan(group, inferencev1alpha1.AFDAttention, 0)
	}
	return vllmconfig.BuildLaunchPlan(group.Spec)
}

// desiredAFDMember reuses the common runtime template with role resources and a non-restarting Pod.
func (reconciler *ModelGroupReconciler) desiredAFDMember(ctx context.Context, group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole, index int32) (*corev1.Pod, error) {
	plan, err := afdMemberLaunchPlan(group, role, index)
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
	template.Labels[afdExecutionRoleLabel], template.Labels[afdMemberIndexLabel] = string(role), strconv.Itoa(int(index))
	template.Spec.RestartPolicy = corev1.RestartPolicyNever
	// Never applies to probe-triggered exits too; no member may restart into this domain.
	template.Spec.Containers[0].Env = append(template.Spec.Containers[0].Env, corev1.EnvVar{
		Name: "FORETOKEN_MEMBER_IP", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}},
	})
	if endpoint := plan.AFD.Coordination; endpoint != nil {
		template.Spec.Containers[0].Ports = append(template.Spec.Containers[0].Ports,
			corev1.ContainerPort{Name: "role-master", ContainerPort: endpoint.MasterPort, Protocol: corev1.ProtocolTCP},
			corev1.ContainerPort{Name: "role-handshake", ContainerPort: endpoint.HandshakePort, Protocol: corev1.ProtocolTCP},
		)
	}
	if role == inferencev1alpha1.AFDFFN {
		template.Spec.Containers[0].Ports = append(template.Spec.Containers[0].Ports, corev1.ContainerPort{Name: "afd", ContainerPort: group.Spec.AFD.Port, Protocol: corev1.ProtocolTCP})
	}
	if err := placeRuntimeCache(ctx, reconciler.Client, group.Namespace, group.Spec.Artifacts.Cache, template); err != nil {
		return nil, err
	}
	if template.Spec.Affinity != nil && template.Spec.Affinity.PodAffinity != nil &&
		(group.Spec.AFD.Attention.MemberCount > 1 || group.Spec.AFD.FFN.MemberCount > 1) {
		for _, term := range template.Spec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution {
			claimUID := template.Labels[runtimeCacheClaimLabel]
			if claimUID != "" && term.TopologyKey == corev1.LabelHostname && term.LabelSelector != nil && term.LabelSelector.MatchLabels[runtimeCacheClaimLabel] == claimUID {
				return nil, fmt.Errorf("multi-member AFD requires a ReadWriteMany runtime cache; single-node cache affinity conflicts with role spreading")
			}
		}
	}
	placeAFDMember(group, role, template)
	pod := &corev1.Pod{ObjectMeta: template.ObjectMeta, Spec: template.Spec}
	pod.Name, pod.Namespace = afdMemberName(group, role, index), group.Namespace
	// Retain the Pod even if deletion races the Group status write. The same owner records its
	// identity and real termination before releasing it; absence never authorizes a replacement.
	controllerutil.AddFinalizer(pod, afdMemberFinalizer)
	if err := controllerutil.SetControllerReference(group, pod, reconciler.Scheme()); err != nil {
		return nil, err
	}
	return pod, nil
}

// placeAFDMember spreads one role per host within its UID and prefers opposite-role peers.
// Cache constraints stay authoritative; soft affinity never establishes a communication pairing.
func placeAFDMember(group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole, pod *corev1.PodTemplateSpec) {
	opposite := inferencev1alpha1.AFDAttention
	if role == inferencev1alpha1.AFDAttention {
		opposite = inferencev1alpha1.AFDFFN
	}
	if pod.Spec.Affinity == nil {
		pod.Spec.Affinity = &corev1.Affinity{}
	}
	pod.Spec.Affinity.PodAntiAffinity = &corev1.PodAntiAffinity{RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{{
		TopologyKey:   corev1.LabelHostname,
		LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{modelGroupUIDLabel: string(group.UID), afdExecutionRoleLabel: string(role)}},
	}}}
	if pod.Spec.Affinity.PodAffinity == nil {
		pod.Spec.Affinity.PodAffinity = &corev1.PodAffinity{}
	}
	pod.Spec.Affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution = []corev1.WeightedPodAffinityTerm{{
		Weight: 100, PodAffinityTerm: corev1.PodAffinityTerm{
			TopologyKey:   corev1.LabelHostname,
			LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{modelGroupUIDLabel: string(group.UID), afdExecutionRoleLabel: string(opposite)}},
		},
	}}
}

// afdDiscoveryServices describes the Group-owned startup endpoints for creation and cleanup.
// F0 always exposes shared rendezvous; multi-member roles add separate coordination ports.
func afdDiscoveryServices(group *inferencev1alpha1.ModelGroup) []*corev1.Service {
	var services []*corev1.Service
	for _, role := range []inferencev1alpha1.AFDExecutionRole{inferencev1alpha1.AFDAttention, inferencev1alpha1.AFDFFN} {
		var ports []corev1.ServicePort
		if role == inferencev1alpha1.AFDFFN {
			ports = append(ports, corev1.ServicePort{Name: "afd", Port: group.Spec.AFD.Port, TargetPort: intstr.FromString("afd"), Protocol: corev1.ProtocolTCP})
		}
		if endpoint := afdRoleCoordination(group, role); endpoint != nil {
			ports = append(ports,
				corev1.ServicePort{Name: "role-master", Port: endpoint.MasterPort, TargetPort: intstr.FromString("role-master"), Protocol: corev1.ProtocolTCP},
				corev1.ServicePort{Name: "role-handshake", Port: endpoint.HandshakePort, TargetPort: intstr.FromString("role-handshake"), Protocol: corev1.ProtocolTCP},
			)
		}
		if len(ports) == 0 {
			continue
		}
		services = append(services, &corev1.Service{
			TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Service"},
			ObjectMeta: metav1.ObjectMeta{Name: afdDiscoveryName(group, role), Namespace: group.Namespace},
			Spec: corev1.ServiceSpec{ClusterIP: corev1.ClusterIPNone, PublishNotReadyAddresses: true,
				Selector: map[string]string{modelGroupUIDLabel: string(group.UID), afdExecutionRoleLabel: string(role), afdMemberIndexLabel: "0"}, Ports: ports},
		})
	}
	return services
}

// reconcileAFDDiscovery owns readiness-independent role and shared F0 discovery for this UID.
func (reconciler *ModelGroupReconciler) reconcileAFDDiscovery(ctx context.Context, group *inferencev1alpha1.ModelGroup) error {
	for _, desired := range afdDiscoveryServices(group) {
		if err := controllerutil.SetControllerReference(group, desired, reconciler.Scheme()); err != nil {
			return err
		}
		current := new(corev1.Service)
		err := reconciler.APIReader.Get(ctx, client.ObjectKeyFromObject(desired), current)
		if err == nil && !metav1.IsControlledBy(current, group) {
			return fmt.Errorf("AFD discovery %q belongs to another owner", current.Name)
		}
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		if err := reconciler.Patch(ctx, desired, client.Apply, client.FieldOwner(modelGroupFieldOwner), client.ForceOwnership); err != nil {
			return err
		}
	}
	return nil
}

// observeAFDMember reads the API server directly so informer lag cannot become a lost member.
// The persisted UID and failure are retained even when a Pod has disappeared.
func (reconciler *ModelGroupReconciler) observeAFDMember(ctx context.Context, group *inferencev1alpha1.ModelGroup, role inferencev1alpha1.AFDExecutionRole, index int32) (*corev1.Pod, *inferencev1alpha1.ModelGroupAFDMemberStatus, error) {
	var member *inferencev1alpha1.ModelGroupAFDMemberStatus
	for i := range group.Status.AFDMembers {
		if group.Status.AFDMembers[i].Role == role && group.Status.AFDMembers[i].MemberIndex == index {
			member = &group.Status.AFDMembers[i]
			break
		}
	}
	pod := new(corev1.Pod)
	err := reconciler.APIReader.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: afdMemberName(group, role, index)}, pod)
	if apierrors.IsNotFound(err) {
		if member != nil && group.DeletionTimestamp.IsZero() && member.FailureReason == "" {
			member.FailureReason, member.FailureMessage = "MemberMissing", fmt.Sprintf("%s/%d Pod %s disappeared; termination is not inferred", role, index, member.PodUID)
		}
		return nil, member, nil
	}
	if err != nil {
		return nil, member, err
	}
	if !metav1.IsControlledBy(pod, group) || pod.Labels[modelGroupUIDLabel] != string(group.UID) || pod.Labels[afdExecutionRoleLabel] != string(role) || pod.Labels[afdMemberIndexLabel] != strconv.Itoa(int(index)) {
		return nil, member, fmt.Errorf("AFD Pod %q has an unexpected owner or member identity", pod.Name)
	}
	if member == nil {
		group.Status.AFDMembers = append(group.Status.AFDMembers, inferencev1alpha1.ModelGroupAFDMemberStatus{Role: role, MemberIndex: index, PodUID: string(pod.UID)})
		member = &group.Status.AFDMembers[len(group.Status.AFDMembers)-1]
	}
	if member.PodUID != string(pod.UID) {
		if !group.DeletionTimestamp.IsZero() {
			return nil, member, fmt.Errorf("AFD Pod %q has a replacement UID; cleanup requires owner intervention", pod.Name)
		}
		member.FailureReason, member.FailureMessage = "MemberReplaced", fmt.Sprintf("%s/%d no longer has recorded Pod UID %s; replacement is not adopted", role, index, member.PodUID)
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

// reconcileAFD creates only never-created members and publishes the complete Group's serving state.
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
	members := afdMembers(group)
	for _, expected := range members {
		role, index := expected.role, expected.index
		if _, _, err := reconciler.observeAFDMember(ctx, group, role, index); err != nil {
			state := modelGroupMaterializedState(false, schedulingNotEvaluated())
			state.unavailable = &modelGroupConditionState{status: metav1.ConditionFalse, reason: "MemberObservationFailed", message: fmt.Sprintf("%s/%d: %v", role, index, err)}
			return ctrl.Result{}, errors.Join(err, reconciler.publishAFDState(ctx, group, pool, state))
		}
	}
	memberStatusErr := reconciler.saveAFDMemberStatus(ctx, group, base)
	for _, member := range group.Status.AFDMembers {
		if member.FailureReason != "" {
			state := modelGroupMaterializedState(false, schedulingNotEvaluated())
			state.phase = inferencev1alpha1.ModelGroupPhaseFailed
			state.unavailable = &modelGroupConditionState{status: metav1.ConditionFalse, reason: member.FailureReason, message: fmt.Sprintf("%s/%d: %s", member.Role, member.MemberIndex, member.FailureMessage)}
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
	if err := reconciler.reconcileAFDDiscovery(ctx, group); err != nil {
		return ctrl.Result{}, err
	}
	var creationErr error
	for _, expected := range members {
		role, index := expected.role, expected.index
		exists := false
		for _, member := range group.Status.AFDMembers {
			if member.Role == role && member.MemberIndex == index {
				exists = true
				break
			}
		}
		if exists {
			continue
		}
		desired, err := reconciler.desiredAFDMember(ctx, group, role, index)
		if err == nil {
			err = reconciler.Create(ctx, desired)
		}
		if err != nil {
			// A failed create must not withhold the other role's independent startup.
			creationErr = errors.Join(creationErr, fmt.Errorf("create %s/%d: %w", role, index, err))
			continue
		}
		base = group.DeepCopy()
		group.Status.AFDMembers = append(group.Status.AFDMembers, inferencev1alpha1.ModelGroupAFDMemberStatus{Role: role, MemberIndex: index, PodUID: string(desired.UID)})
		if err := reconciler.saveAFDMemberStatus(ctx, group, base); err != nil {
			return ctrl.Result{}, errors.Join(creationErr, err)
		}
	}
	if creationErr != nil {
		return ctrl.Result{}, creationErr
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
		pending := false
		for index := int32(0); index < afdRoleSpec(group, role).MemberCount; index++ {
			base := group.DeepCopy()
			pod, member, err := reconciler.observeAFDMember(ctx, group, role, index)
			if err != nil {
				return ctrl.Result{}, err
			}
			if err := reconciler.saveAFDMemberStatus(ctx, group, base); err != nil {
				return ctrl.Result{}, err
			}
			if pod == nil {
				if member != nil && !member.TerminationConfirmed {
					pending = true
				}
				continue
			}
			pending = true
			if pod.DeletionTimestamp.IsZero() {
				if err := reconciler.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}); client.IgnoreNotFound(err) != nil {
					return ctrl.Result{}, err
				}
				continue
			}
			if member.TerminationConfirmed && controllerutil.ContainsFinalizer(pod, afdMemberFinalizer) {
				before := pod.DeepCopy()
				controllerutil.RemoveFinalizer(pod, afdMemberFinalizer)
				if err := reconciler.Patch(ctx, pod, client.MergeFrom(before)); err != nil {
					return ctrl.Result{}, err
				}
			}
		}
		if pending {
			return ctrl.Result{RequeueAfter: modelGroupDrainPollInterval}, reconciler.updateDrainStatus(ctx, group, group.Status.DrainStartedAt, false, false, false, "TerminationUnconfirmed", fmt.Sprintf("Waiting for all %s members to terminate and disappear", role))
		}
	}
	for _, desired := range afdDiscoveryServices(group) {
		service := new(corev1.Service)
		err := reconciler.APIReader.Get(ctx, client.ObjectKeyFromObject(desired), service)
		if err == nil {
			if !metav1.IsControlledBy(service, group) {
				return ctrl.Result{}, fmt.Errorf("AFD discovery %q belongs to another owner", service.Name)
			}
			if err := reconciler.Delete(ctx, service, client.Preconditions{UID: &service.UID}); client.IgnoreNotFound(err) != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{RequeueAfter: modelGroupDrainPollInterval}, nil
		}
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
	}
	return ctrl.Result{}, reconciler.removeDrainFinalizer(ctx, group)
}
