// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

package v1alpha1

// AFDParallelism is the authoritative role-local rank layout, independent of replicas.
type AFDParallelism struct {
	// +kubebuilder:validation:Minimum=1
	DP int32 `json:"dp"`
	// +kubebuilder:validation:Minimum=1
	TP int32 `json:"tp"`
}

// AFDPoolRole requests one homogeneous member per role node.
// +kubebuilder:validation:XValidation:rule="self.nodes * self.resources.requests.gpu.count == self.parallelism.dp * self.parallelism.tp",message="role accelerator capacity must equal DP times TP"
type AFDPoolRole struct {
	// +kubebuilder:validation:Minimum=1
	Nodes       int32          `json:"nodes"`
	Resources   ModelResources `json:"resources"`
	Parallelism AFDParallelism `json:"parallelism"`
}

// AFDPool describes one complete instance in the user-owned Aggregate Pool.
type AFDPool struct {
	Attention AFDPoolRole `json:"attention"`
	FFN       AFDPoolRole `json:"ffn"`
}

// AFDMemberSpec fixes resource and rank ownership for all members of one role.
// +kubebuilder:validation:XValidation:rule="self.memberCount * self.resources.requests.gpu.count == self.parallelism.dp * self.parallelism.tp",message="role accelerator capacity must equal DP times TP"
type AFDMemberSpec struct {
	// +kubebuilder:validation:Minimum=1
	MemberCount int32          `json:"memberCount"`
	Resources   ModelResources `json:"resources"`
	Parallelism AFDParallelism `json:"parallelism"`
}

// AFDTopology is the normalized complete topology retained by the Pool and Group.
type AFDTopology struct {
	Attention AFDMemberSpec `json:"attention"`
	FFN       AFDMemberSpec `json:"ffn"`
}

// ModelGroupAFD pins static transport settings; UID and endpoints belong to materialization.
type ModelGroupAFD struct {
	AFDTopology `json:",inline"`
	// +kubebuilder:validation:Enum=P2pNcclAFDConnector
	Connector string `json:"connector"`
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	Port int32 `json:"port"`
}

// AFDExecutionRole identifies a member inside an aggregate Group, not a service role.
type AFDExecutionRole string

const (
	AFDAttention AFDExecutionRole = "attention"
	AFDFFN       AFDExecutionRole = "ffn"
)
