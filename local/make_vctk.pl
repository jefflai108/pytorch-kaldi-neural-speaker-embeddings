#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#
# Usage: make_voxceleb1.pl /export/voxceleb1 data/

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb1> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb1 data/\n";
  exit(1);
}

($data_base, $out_dir) = @ARGV;
my $out_vctk_dir = "$out_dir/vctk";

if (system("mkdir -p $out_vctk_dir") != 0) {
  die "Error making directory $out_vctk_dir";
}

opendir my $dh, "$data_base/wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

open(SPKR_VCTK, ">", "$out_vctk_dir/utt2spk") or die "Could not open the output file $out_vctk_dir/utt2spk";
open(WAV_VCTK, ">", "$out_vctk_dir/wav.scp") or die "Could not open the output file $out_vctk_dir/wav.scp";

foreach (@spkr_dirs) {
  my $spkr_id = $_;
  my $new_spkr_id = $spkr_id;
  # If we're using a newer version of VoxCeleb1, we need to "deanonymize"
  # the speaker labels.
  if (exists $id2spkr{$spkr_id}) {
    $new_spkr_id = $id2spkr{$spkr_id};
  }
  opendir my $dh, "$data_base/wav/$spkr_id/" or die "Cannot open directory: $!";
  my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
  closedir $dh;
  foreach (@files) {
    my $filename = $_;
    my $rec_id = substr($filename, 0, 4);
    my $segment = substr($filename, 5, 3);
    my $wav = "$data_base/wav/$spkr_id/$filename.wav";
    my $utt_id = "$new_spkr_id-$rec_id-$segment";

    print WAV_VCTK "$utt_id", " $wav", "\n";
    print SPKR_VCTK "$utt_id", " $new_spkr_id", "\n";
  }
}

close(SPKR_VCTK) or die;
close(WAV_VCTK) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $out_vctk_dir/utt2spk >$out_vctk_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_vctk_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $out_vctk_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $out_vctk_dir") != 0) {
  die "Error validating directory $out_vctk_dir";
}
