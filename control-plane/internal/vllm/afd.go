// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

package vllm

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	inferencev1alpha1 "github.com/shiweijiezero/foretoken/control-plane/api/v1alpha1"
)

// AFDConnector is the transport implemented by the pinned CUDA runtime adapter.
const AFDConnector = "P2pNcclAFDConnector"

// LaunchAFDPlan is the V2 member contract consumed by the managed Rust runtime.
type LaunchAFDPlan struct {
	Role              inferencev1alpha1.AFDExecutionRole `json:"role"`
	MemberIndex       int32                              `json:"memberIndex"`
	MemberCount       int32                              `json:"memberCount"`
	GroupUID          string                             `json:"groupUid"`
	Host              string                             `json:"host"`
	Port              int32                              `json:"port"`
	NumAttentionRanks int32                              `json:"numAttentionRanks"`
	NumFFNRanks       int32                              `json:"numFfnRanks"`
	Coordination      *AFDCoordination                   `json:"coordination,omitempty"`
}

// AFDCoordination locates member zero of one role before application readiness.
// The Group materializer owns its UID-scoped discovery and distinct listening ports.
type AFDCoordination struct {
	Host          string `json:"host"`
	MasterPort    int32  `json:"masterPort"`
	HandshakePort int32  `json:"handshakePort"`
}

// AFDMember supplies identities owned by Group materialization, never by its immutable spec.
type AFDMember struct {
	Role           inferencev1alpha1.AFDExecutionRole
	Index          int32
	GroupUID       string
	RendezvousHost string
	Coordination   *AFDCoordination
}

// validateAFD enforces the pinned plugin and current managed runtime's execution limits.
// Generic resource identities remain the compiler's responsibility.
func validateAFD(template inferencev1alpha1.NormalizedPoolTemplate, args inferencev1alpha1.EngineArguments) error {
	if template.Backend != "vllm" || template.Role != inferencev1alpha1.ModelRoleAggregate ||
		template.NodeCount != 0 || template.Resources != nil || template.KVCache != nil ||
		template.ECProfile != "" || template.Profiling != nil {
		return fmt.Errorf("AFD requires aggregate vLLM, role-local topology, and no KV/EC transfer or diagnostic profiling")
	}
	if err := validateAFDTopology(*template.AFD, template.MemberCount); err != nil {
		return err
	}
	// The current workload owner still materializes exactly one single-GPU Pod per role.
	for _, side := range []inferencev1alpha1.AFDMemberSpec{template.AFD.Attention, template.AFD.FFN} {
		if side.MemberCount != 1 || side.Resources.Requests.GPU.Count != 1 {
			return fmt.Errorf("AFD workload materialization currently requires one single-GPU member per role")
		}
	}
	ownedOptions := []string{
		"tensor-parallel-size", "pipeline-parallel-size", "data-parallel-size",
		"prefill-context-parallel-size", "decode-context-parallel-size",
		"enable-expert-parallel", "enable-eplb", "all2all-backend",
		"worker-extension-cls", "enable-fault-tolerance", "enable-dbo",
	}
	names := make([]string, 0, len(args))
	for name := range args {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		value := args[name]
		for _, owned := range ownedOptions {
			if strings.HasPrefix(owned, name) {
				return fmt.Errorf("engineArgs.%s conflicts with the managed AFD plan", name)
			}
		}
		if strings.HasPrefix("enforce-eager", name) && (name != "enforce-eager" || string(value.Raw) != "true") {
			return fmt.Errorf("managed AFD requires enforce-eager=true")
		}
		if strings.HasPrefix("additional-config", name) {
			if name != "additional-config" {
				return fmt.Errorf("AFD requires the full additional-config option name")
			}
			if err := validateAdditionalConfig(value.Raw, true); err != nil {
				return err
			}
		}
		if name == "load-format" {
			var loader string
			if json.Unmarshal(value.Raw, &loader) == nil && loader == "modelexpress" {
				return fmt.Errorf("AFD does not support ModelExpress weight sources")
			}
		}
	}
	return nil
}

// validateAFDTopology verifies the normalized projection before the adapter consumes it.
func validateAFDTopology(topology inferencev1alpha1.AFDTopology, members int32) error {
	if int64(members) != int64(topology.Attention.MemberCount)+int64(topology.FFN.MemberCount) {
		return fmt.Errorf("AFD member count must equal both role member counts")
	}
	for _, side := range []inferencev1alpha1.AFDMemberSpec{topology.Attention, topology.FFN} {
		p := side.Parallelism
		ranks := int64(p.DP) * int64(p.TP)
		gpus := side.Resources.Requests.GPU.Count
		if side.MemberCount < 1 || gpus < 1 || p.DP < 1 || p.TP < 1 || ranks > math.MaxInt32 ||
			int64(side.MemberCount)*int64(gpus) != ranks {
			return fmt.Errorf("AFD requires positive role resources with memberCount times GPUs equal to DP times TP")
		}
		if gpus%p.TP != 0 && p.TP%gpus != 0 {
			return fmt.Errorf("AFD TP must divide local GPUs or span an integral number of members")
		}
	}
	if int64(topology.Attention.Parallelism.DP)*int64(topology.Attention.Parallelism.TP) <
		int64(topology.FFN.Parallelism.DP)*int64(topology.FFN.Parallelism.TP) {
		return fmt.Errorf("P2pNcclAFDConnector requires Attention ranks >= FFN ranks")
	}
	return nil
}

// validateAdditionalConfig protects platform-owned AFD configuration without rewriting native values.
func validateAdditionalConfig(raw []byte, requireObject bool) error {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return fmt.Errorf("invalid additional-config JSON: %w", err)
	}
	if encoded, ok := value.(string); ok {
		if err := json.Unmarshal([]byte(encoded), &value); err != nil {
			return fmt.Errorf("invalid additional-config JSON: %w", err)
		}
	}
	object, ok := value.(map[string]any)
	if !ok {
		if requireObject {
			return fmt.Errorf("AFD additional-config must be a JSON object")
		}
		return nil
	}
	if _, owned := object["afd"]; owned {
		return fmt.Errorf("engineArgs.additional-config.afd is owned by Foretoken")
	}
	return nil
}

// BuildAFDLaunchPlan projects one materialized member for Rust; it owns no discovery resources.
// The caller provides the actual Group UID and F0 address after the Group has been persisted.
func BuildAFDLaunchPlan(group inferencev1alpha1.ModelGroupSpec, member AFDMember) (LaunchPlan, error) {
	afd := group.AFD
	if afd == nil || group.NodeCount != 0 || group.Resources != nil || group.Parallelism != nil ||
		group.Runtime.Backend != "vllm" || group.Role != inferencev1alpha1.ModelRoleAggregate ||
		group.KVRuntime != nil || group.PDRuntime != nil || group.ECRuntime != nil || group.Runtime.Profiling != nil {
		return LaunchPlan{}, fmt.Errorf("AFD member plan requires a complete aggregate AFD Group")
	}
	if err := validateAFDTopology(afd.AFDTopology, group.MemberCount); err != nil {
		return LaunchPlan{}, err
	}
	if afd.Connector != AFDConnector || afd.Port < 1 || afd.Port > 65535 ||
		member.GroupUID == "" || member.RendezvousHost == "" {
		return LaunchPlan{}, fmt.Errorf("AFD requires its supported connector, port, Group UID and F0 address")
	}
	var side inferencev1alpha1.AFDMemberSpec
	switch member.Role {
	case inferencev1alpha1.AFDAttention:
		side = afd.Attention
	case inferencev1alpha1.AFDFFN:
		side = afd.FFN
	default:
		return LaunchPlan{}, fmt.Errorf("unknown AFD execution role %q", member.Role)
	}
	if member.Index < 0 || member.Index >= side.MemberCount {
		return LaunchPlan{}, fmt.Errorf("AFD member index is outside its role")
	}
	if side.MemberCount > 1 && member.Coordination == nil {
		return LaunchPlan{}, fmt.Errorf("multi-member AFD requires a role coordination endpoint")
	}
	if endpoint := member.Coordination; endpoint != nil {
		if endpoint.Host == "" || endpoint.MasterPort < 1 || endpoint.MasterPort > 65535 ||
			endpoint.HandshakePort < 1 || endpoint.HandshakePort > 65535 || endpoint.MasterPort == endpoint.HandshakePort {
			return LaunchPlan{}, fmt.Errorf("AFD role coordination requires a host and distinct valid master/handshake ports")
		}
	}
	plan, err := buildLaunchPlan(group, AFDMemberParallelism(side))
	if err != nil {
		return LaunchPlan{}, err
	}
	plan.Version = 2
	plan.AFD = &LaunchAFDPlan{
		Role: member.Role, MemberIndex: member.Index, MemberCount: side.MemberCount,
		GroupUID: member.GroupUID, Host: member.RendezvousHost, Port: afd.Port,
		NumAttentionRanks: afd.Attention.Parallelism.DP * afd.Attention.Parallelism.TP,
		NumFFNRanks:       afd.FFN.Parallelism.DP * afd.FFN.Parallelism.TP,
	}
	if member.Coordination != nil {
		endpoint := *member.Coordination
		plan.AFD.Coordination = &endpoint
	}
	return plan, nil
}

// AFDMemberParallelism gives launch and serving consumers the same role-local layout.
// The pinned adapter owns the fixed context/pipeline dimensions and required expert parallelism.
func AFDMemberParallelism(side inferencev1alpha1.AFDMemberSpec) inferencev1alpha1.CompiledParallelism {
	p := side.Parallelism
	return inferencev1alpha1.CompiledParallelism{
		TP: p.TP, DP: p.DP, PP: 1, PCP: 1, DCP: 1,
		EP: &inferencev1alpha1.ExpertParallelism{Size: p.DP * p.TP},
	}
}
